with open("/mnt/f/mambacd/home/HICD_v7/script/train_full_v7.py", "r") as f:
    content = f.read()

# Patch 1: Add set_phase(1) after freezing backbone
old1 = """            else:
                self.model.freeze_backbone()

        t_start = time.time()"""

new1 = """            else:
                self.model.freeze_backbone()
            # Phase 1: rare-class focused weights
            if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'set_phase'):
                self.model.loss_fn.set_phase(1)

        t_start = time.time()"""

content = content.replace(old1, new1)

# Patch 2: Add set_phase(2) after unfreezing
old2 = """                else:
                    self.model.unfreeze_backbone(self.args.unfreeze_ratio)
            train_loss, train_dict = self.train_epoch(epoch)"""

new2 = """                else:
                    self.model.unfreeze_backbone(self.args.unfreeze_ratio)
                # Phase 2: restore normal weights
                if hasattr(self.model, 'loss_fn') and hasattr(self.model.loss_fn, 'set_phase'):
                    self.model.loss_fn.set_phase(2, self.dataset_config)
            train_loss, train_dict = self.train_epoch(epoch)"""

content = content.replace(old2, new2)

with open("/mnt/f/mambacd/home/HICD_v7/script/train_full_v7.py", "w") as f:
    f.write(content)

with open("/mnt/f/mambacd/home/HICD_v7/script/train_full_v7.py", "r") as f:
    c = f.read()
print("set_phase(1) found:", "set_phase(1)" in c)
print("set_phase(2 found:", "set_phase(2" in c)