"""Use the explicit corrected builder when evaluating corrected-run checkpoints."""
import eval_epoch50_strict_test as evaluation
from models.dino_corrected_model import build_dino_corrected_hicd

if __name__ == '__main__':
    evaluation.build_hicd_v7 = build_dino_corrected_hicd
    evaluation.main()
