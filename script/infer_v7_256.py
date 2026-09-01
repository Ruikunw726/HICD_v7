import os, sys, yaml, torch, numpy as np
sys.path.insert(0, '.')
from models import build_hicd_v7
from datasets import create_dataloader
from evaluation import ICDEvaluatorV6 as ICDEvaluator
import torch.nn.functional as F

with open("datasets/configs/second.yaml", "r", encoding="utf-8") as f:
    dataset_config = yaml.safe_load(f)
model_config = dict(dataset_config)
model_config["backbone"] = {"variant": "L1", "pretrained": False, "pretrained_path": None}
model_config["hidden_dim"] = 128
model_config["active_heads"] = dataset_config.get("active_heads", {})
model_config["pixel_class_weights"] = [2.931, 0.160, 0.061, 2.452, 0.090, 0.307]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_hicd_v7(model_config).to(device)
ckpt = torch.load("checkpoints/v7_second_256_v2/checkpoint_epoch_70.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded epoch {ckpt.get('epoch','?')}")

val_loader = create_dataloader(data_dir="/mnt/c/SECOND_256", config=dataset_config, split="val", batch_size=4, num_workers=0, img_size=256)

evaluator = ICDEvaluator(dataset_config)

with torch.no_grad():
    for batch in val_loader:
        img_t1 = batch["img_t1"].to(device)
        img_t2 = batch["img_t2"].to(device)
        targets = {}
        label = batch["label"].to(device)
        if label.dim() == 3:
            label_s = label.unsqueeze(1).float()
            label_s = F.interpolate(label_s, scale_factor=0.25, mode="nearest").squeeze(1).long()
            targets["target_gt"] = label_s
        cs = batch["change_state"].to(device)
        if cs.dim() == 3:
            cs_s = cs.unsqueeze(1).float()
            cs_s = F.interpolate(cs_s, scale_factor=0.25, mode="nearest").squeeze(1).long()
            targets["change_state"] = cs_s
        instance_labels = []
        instance_classes = batch.get("instance_classes")
        instance_states = batch.get("instance_states")
        for i, boxes in enumerate(batch["instance_boxes"]):
            if boxes is not None and len(boxes) > 0:
                labels = (instance_classes[i].to(device) if instance_classes is not None and instance_classes[i] is not None else torch.zeros(len(boxes), dtype=torch.long, device=device))
                inst_dict = {"labels": labels, "boxes": boxes.to(device)}
                if instance_states is not None and instance_states[i] is not None:
                    inst_dict["states"] = instance_states[i].to(device)
                instance_labels.append(inst_dict)
            else:
                instance_labels.append({"labels": torch.zeros(0, dtype=torch.long, device=device), "boxes": torch.zeros(0, 4, device=device)})
        targets["instance_labels"] = instance_labels
        predictions, loss, loss_dict = model(img_t1, img_t2, targets)
        evaluator.update(predictions, targets, loss_dict)

results = evaluator.compute()
print(evaluator.format_results(results))
