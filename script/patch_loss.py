
import torch

with open('/mnt/f/mambacd/home/HICD_v7/models/losses/v7_loss.py', 'r') as f:
    content = f.read()

old_text = '''    def get_class_weights(self):
        """Get current adaptive weights for logging."""
        return self.adaptive_weighter.get_weights()'''

new_text = '''    def get_class_weights(self):
        """Get current adaptive weights for logging."""
        return self.adaptive_weighter.get_weights()

    def set_phase(self, phase, config=None):
        """Switch between training phases with different class weights."""
        if phase == 1:
            pw = [0.0, 10.0, 10.0, 5.0, 50.0, 5.0, 0.0, 50.0, 50.0, 50.0, 10.0, 10.0, 50.0, 20.0, 10.0]
            sw = [0.0, 5.0, 8.0, 8.0, 8.0, 8.0, 0.0]
            print("[Phase 1] Rare-class focused weights activated")
        else:
            if config:
                pw = config.get("pixel_class_weights", None)
                sw = config.get("state_pixel_weights", None)
            else:
                pw = None
                sw = None
            print("[Phase 2] Normal class weights restored")
        if pw is not None:
            self.adaptive_weighter.weights = torch.tensor(pw, dtype=torch.float32)
        if sw is not None:
            self.state_weighter.weights = torch.tensor(sw, dtype=torch.float32)'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('/mnt/f/mambacd/home/HICD_v7/models/losses/v7_loss.py', 'w') as f:
        f.write(content)
    print("Patched successfully")
else:
    print("ERROR: target text not found")
