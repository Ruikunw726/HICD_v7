"""Compact presentation only; never round scores used for training/selection."""
from __future__ import annotations


def metric_number(value):
    return "n/a" if value is None else f"{value:.4f}"


def format_strict_results(result):
    lines = [f"[{result['protocol_version']}] joint micro F1={metric_number(result['strict_joint_f1'])} "
             f"macro F1={metric_number(result['strict_joint_macro_f1'])}"]
    tasks = [f"{name}: {metric_number(value['strict_joint_f1'])}/{metric_number(value['strict_joint_macro_f1'])}"
             for name, value in result['tasks'].items()
             if value['strict_joint_micro']['support'] + value['strict_joint_micro']['predicted_support'] > 0]
    if tasks:
        lines.append("Task micro/macro  " + "  ".join(tasks))
    lines.append("Target         joint_P  joint_R joint_F1   loc_F1 state_F1")
    for target in result['per_target'].values():
        micro = target['strict_joint_micro']
        if micro['support'] + micro['predicted_support'] == 0:
            continue
        values = (micro['precision'], micro['recall'], micro['f1'],
                  target['target_localization_f1'], target['state_on_gt_target']['micro']['f1'])
        lines.append(f"{target['target_name']:<14}" + " ".join(f"{metric_number(value):>8}" for value in values))
    return "\n".join(lines)


def format_epoch_summary(artifact):
    lines = [f"Epoch {artifact['epoch']} | train loss={metric_number(artifact['train_loss'])} "
             f"val loss={metric_number(artifact['validation_loss'])} "
             f"best joint F1={metric_number(artifact['best_score'])}",
             format_strict_results(artifact['validation'])]
    return "\n".join(lines)


def format_csv_row(row):
    """Fixed four decimals for scalar floats, integer counts kept as integers."""
    return {key: metric_number(value) if isinstance(value, float) else value
            for key, value in row.items()}
