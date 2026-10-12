"""CPU-only strict v1 evaluation of decoded prediction NPZ shards.

No model imports, no GPU use, no modification of training logs. Each shard
must contain integer pred_target/pred_state/gt_target/gt_state arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Import directly: evaluation/__init__.py imports legacy torch/scipy modules.
sys.path.insert(0, str(ROOT / "evaluation"))
from strict_joint_protocol import StrictJointEvaluator
from strict_joint_reporting import format_strict_results


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_cache(paths):
    evaluator = StrictJointEvaluator()
    sources = []
    required = ("pred_target", "pred_state", "gt_target", "gt_state")
    seen = set()
    for path in paths:
        path = Path(path).resolve()
        if path in seen:
            raise ValueError(f"Duplicate NPZ path: {path}")
        seen.add(path)
        before = sha256(path)
        with np.load(path, allow_pickle=False) as shard:
            missing = set(required) - set(shard.files)
            if missing:
                raise ValueError(f"{path}: missing arrays {sorted(missing)}")
            evaluator.update(*(shard[key] for key in required))
        if before != sha256(path):
            raise RuntimeError(f"Cache changed during evaluation: {path}")
        sources.append({"path": str(path), "sha256": before})
    if not sources:
        raise ValueError("No NPZ prediction shards found")
    return evaluator, sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="NPZ files or directories of NPZ shards, in deterministic order")
    parser.add_argument("--output", required=True, help="New JSON path; overwrite is refused")
    parser.add_argument("--split", choices=("val", "test", "unspecified"), default="unspecified")
    parser.add_argument("--provenance", help="Optional JSON: checkpoint, GT policy, resolution, code version")
    options = parser.parse_args(argv)
    output = Path(options.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    paths = []
    for name in options.predictions:
        path = Path(name)
        paths.extend(sorted(path.glob("*.npz")) if path.is_dir() else [path])
    evaluator, sources = evaluate_cache(paths)
    result = evaluator.compute()
    provenance = json.loads(Path(options.provenance).read_text(encoding="utf-8")) if options.provenance else {}
    if not isinstance(provenance, dict):
        raise ValueError("Provenance must be a JSON object")
    result["provenance"] = {"producer": provenance, "sources": sources,
                            "split": options.split, "mode": "cached_decoded_arrays_cpu",
                            "coverage": "provided_shards_only; dataset completeness not verified"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(format_strict_results(result))
    print(f"valid_pixels={result['valid_pixels']} ignored_pixels={result['ignored_pixels']}")
    return result


if __name__ == "__main__":
    main()
