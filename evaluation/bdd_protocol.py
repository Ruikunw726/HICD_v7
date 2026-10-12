"""
Unified evaluation protocol: F_loc / F_clf / F_overall + SeK

Based on WD-CD paper's BDD protocol, extended to all tasks.
"""

import numpy as np


def compute_f1(tp, fp, fn):
    """Compute F1, precision, recall."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1, precision, recall


def compute_sek(state_preds, state_gts, num_states=7):
    """
    Separated Kappa (SeK) for semantic change detection.
    Standard formula from SCD literature:
      IOU1 = q00 / (col0 + row0 - q00)
      IOU2 = sum_{i>=1,j>=1} q_ij / (total - q00)
      rho_hat = trace_changed / N_change
      eta_hat = sum_j(row_hat[j]*col_hat[j]) / N_change^2
      SeK = exp(IOU2 - 1) * (rho_hat - eta_hat) / (1 - eta_hat)

    Args:
        state_preds: predicted state classes
        state_gts: ground truth state classes
        num_states: number of state classes (default 7)

    Returns:
        sek, iou1, iou2, rho_hat, eta_hat
    """
    # Filter invalid states
    valid_mask = (state_gts < num_states) & (state_preds < num_states)
    v_preds = state_preds[valid_mask]
    v_gts = state_gts[valid_mask]

    if len(v_gts) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    # Build confusion matrix Q: rows=GT, cols=Pred
    C = num_states
    Q = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        for j in range(C):
            Q[i, j] = int(((v_gts == i) & (v_preds == j)).sum())

    q00 = Q[0, 0]  # correctly predicted no-change
    total = Q.sum()

    # IOU1: unchanged class IoU
    col0 = Q[:, 0].sum()  # all predicted as 0
    row0 = Q[0, :].sum()  # all GT=0
    iou1_denom = col0 + row0 - q00
    iou1 = q00 / iou1_denom if iou1_denom > 0 else float('nan')

    # IOU2: changed classes IoU
    N_change = total - q00
    changed_correct = Q[1:, 1:].sum()
    iou2 = changed_correct / N_change if N_change > 0 else float('nan')

    # Modified confusion matrix Q_hat: Q_hat[0,0] = 0
    Q_hat = Q.copy()
    Q_hat[0, 0] = 0.0

    # rho_hat: accuracy among changed pixels
    trace_changed = np.trace(Q_hat)
    rho_hat = trace_changed / N_change if N_change > 0 else float('nan')

    # eta_hat: expected agreement under independence on Q_hat
    row_hat = Q_hat.sum(axis=1)
    col_hat = Q_hat.sum(axis=0)
    eta_hat = np.sum(row_hat * col_hat) / (N_change ** 2) if N_change > 0 else float('nan')

    # SeK
    denom = 1.0 - eta_hat
    if N_change == 0 or np.isnan(iou2) or np.isnan(rho_hat) or np.isnan(eta_hat):
        sek = float('nan')
    elif abs(denom) < 1e-12:
        # eta_hat ≈ 1.0: model predictions match random expectation perfectly
        # This happens with perfect prediction on imbalanced data
        sek = 1.0 if rho_hat > 0.99 else 0.0
    else:
        sek = float(np.exp(iou2 - 1.0) * (rho_hat - eta_hat) / denom)

    return sek, iou1, iou2, rho_hat, eta_hat


def evaluate_bdd_protocol(pred_target, pred_state, gt_target, gt_state,
                          target_class, bg_class=15):
    """
    Evaluate a single target class using BDD protocol.
    
    Args:
        pred_target: [N] predicted target classes
        pred_state: [N] predicted state classes
        gt_target: [N] ground truth target classes
        gt_state: [N] ground truth state classes
        target_class: which class to evaluate (e.g., 6 for Building)
        bg_class: background class id (default 15)
    
    Returns:
        dict with F_loc, F_clf, F_overall, SeK, etc.
    """
    # GT mask for this target
    gt_mask = (gt_target == target_class)
    
    if gt_mask.sum() == 0:
        return None
    
    # Pred mask for this target
    pred_mask = (pred_target == target_class)
    
    # F_loc: target vs background
    tp_loc = np.logical_and(pred_mask, gt_mask).sum()
    fp_loc = np.logical_and(pred_mask, ~gt_mask).sum()
    fn_loc = np.logical_and(~pred_mask, gt_mask).sum()
    f1_loc, prec_loc, rec_loc = compute_f1(tp_loc, fp_loc, fn_loc)
    
    # For F_clf, need valid states (exclude replaced=5 and none=6)
    valid_mask = gt_mask & (gt_state != 5) & (gt_state != 6)
    
    if valid_mask.sum() == 0:
        return {
            'F_loc': f1_loc, 'F_clf': 0.0, 'F_overall': 0.3 * f1_loc,
            'F_macro': 0.0, 'SeK': 0.0,
            'prec_loc': prec_loc, 'rec_loc': rec_loc,
        }
    
    # ND = no-change(0), added(3), extended(4)
    # D = damaged(1), reduced(2)
    gt_nd = valid_mask & np.isin(gt_state, [0, 3, 4])
    gt_d = valid_mask & np.isin(gt_state, [1, 2])
    
    pred_nd = pred_mask & np.isin(pred_state, [0, 3, 4])
    pred_d = pred_mask & np.isin(pred_state, [1, 2])
    
    # ND F1
    tp_nd = np.logical_and(pred_nd, gt_nd).sum()
    fp_nd = np.logical_and(pred_nd, gt_d).sum()
    fn_nd = np.logical_and(~pred_nd, gt_nd).sum()
    f1_nd, prec_nd, rec_nd = compute_f1(tp_nd, fp_nd, fn_nd)
    
    # D F1
    tp_d = np.logical_and(pred_d, gt_d).sum()
    fp_d = np.logical_and(pred_d, gt_nd).sum()
    fn_d = np.logical_and(~pred_d, gt_d).sum()
    f1_d, prec_d, rec_d = compute_f1(tp_d, fp_d, fn_d)
    
    # F_clf = harmonic mean of ND and D F1
    f1_clf = 2 * f1_nd * f1_d / max(f1_nd + f1_d, 1e-8)
    
    # F_overall = 0.3 * F_loc + 0.7 * F_clf
    f1_overall = 0.3 * f1_loc + 0.7 * f1_clf
    
    # F_macro = arithmetic mean
    f1_macro = (f1_nd + f1_d) / 2

    # SeK: compute on target pixels only (exclude background target=15 and state=6/none)
    # This reflects true change detection ability within target regions
    sek_valid_mask = gt_mask & (gt_state < 6) & (pred_state < 6)  # 0-5 only, exclude none(6)
    if sek_valid_mask.sum() > 0:
        sek, iou1, iou2, rho_hat, eta_hat = compute_sek(
            pred_state[sek_valid_mask], gt_state[sek_valid_mask], num_states=6
        )
    else:
        sek, iou1, iou2, rho_hat, eta_hat = float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    return {
        'F_loc': f1_loc,
        'F_clf': f1_clf,
        'F_overall': f1_overall,
        'F_macro': f1_macro,
        'F_ND': f1_nd,
        'F_D': f1_d,
        'SeK': sek,
        'iou1': iou1,
        'iou2': iou2,
        'rho_hat': rho_hat,
        'eta_hat': eta_hat,
        'prec_loc': prec_loc,
        'rec_loc': rec_loc,
        'prec_ND': prec_nd,
        'rec_ND': rec_nd,
        'prec_D': prec_d,
        'rec_D': rec_d,
    }


# Task definitions
TASKS = {
    'T1_Building': {
        'name': '建筑损毁',
        'classes': [6],
        'class_names': ['Building'],
    },
    'T2_Surface': {
        'name': '地表覆盖变化',
        'classes': [0, 1, 2, 3, 5],
        'class_names': ['Farmland', 'Runway', 'Taxiway', 'Apron', 'Highway'],
    },
    'T3_Mobile': {
        'name': '移动目标变化',
        'classes': [12, 13],
        'class_names': ['Aircraft', 'Vessel'],
    },
    'T4_Facility': {
        'name': '设施变化',
        'classes': [4, 7, 9, 10, 11, 14],  # Shelter, Pier, Dock, Tank, Crater (Tower removed)
        'class_names': ['Bridge', 'Shelter', 'Pier', 'Dock', 'Tank', 'Crater'],
    },
}

TARGET_NAMES = [
    'Farmland', 'Runway', 'Taxiway', 'Apron', 'Bridge', 'Highway',
    'Building', 'Shelter', 'Tower', 'Pier', 'Dock', 'Tank',
    'Aircraft', 'Vessel', 'Crater', 'Background'
]


def evaluate_four_tasks(pred_target, pred_state, gt_target, gt_state):
    """
    Evaluate all four tasks using BDD protocol + SeK.
    
    Returns:
        dict with results for each task
    """
    results = {}
    
    for task_id, task_info in TASKS.items():
        task_results = {}
        
        # Evaluate each class in this task
        for cls_id, cls_name in zip(task_info['classes'], task_info['class_names']):
            result = evaluate_bdd_protocol(
                pred_target, pred_state, gt_target, gt_state,
                target_class=cls_id, bg_class=15
            )
            if result is not None:
                task_results[cls_name] = result
        
        # Aggregate: use best class for each metric
        if task_results:
            # For single-class tasks (T1), just use that class
            if len(task_info['classes']) == 1:
                cls_name = task_info['class_names'][0]
                task_results['aggregate'] = task_results[cls_name]
            else:
                # For multi-class tasks, compute weighted average
                # Weight by pixel count (approximate by using F_loc as proxy)
                agg = {}
                for metric in ['F_loc', 'F_clf', 'F_overall', 'F_macro', 'SeK']:
                    values = [r[metric] for r in task_results.values() if metric in r]
                    agg[metric] = np.mean(values) if values else 0.0
                task_results['aggregate'] = agg
        
        results[task_id] = task_results
    
    return results
