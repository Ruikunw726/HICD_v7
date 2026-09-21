
with open('/mnt/f/mambacd/home/HICD_v7/models/losses/v7_loss.py', 'r') as f:
    c = f.read()

old_text = '                # Exclude background\n                valid_mask = (target_gt != 255)'

new_text = """                # Exclude background
                valid_mask = (target_gt != 255)
                # Only compute state loss on actual changed pixels (state 1-5)
                # Ignore no-change (0) and none (6) to avoid gradient domination
                state_gt_raw_mask = targets["state_gt"].clamp(0, state_logits.shape[1] - 1)
                changed_mask = (state_gt_raw_mask >= 1) & (state_gt_raw_mask <= 5)
                valid_mask = valid_mask & changed_mask"""

if old_text in c:
    c = c.replace(old_text, new_text)
    with open('/mnt/f/mambacd/home/HICD_v7/models/losses/v7_loss.py', 'w') as f:
        f.write(c)
    print('Patched')
else:
    print('ERROR: target not found')
