#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
cd "$ROOT"
PYTHON=/root/miniconda/envs/mamba/bin/python
EXP=v13_dino_corrected_t123_disjoint_e100_20261012
RUN="$ROOT/checkpoints/$EXP"
SESSION="hicd_$EXP"
ARGS=(--dataset 0617final_t123_dino_corrected_20261012 --config_dir datasets/configs
      --data_dir /mnt/d/CD/0617final_256_v2_disjoint_20261009 --batch_size 32 --img_size 256
      --max_epochs 100 --learning_rate 1e-4 --use_amp --num_workers 4 --backbone_type dinov3
      --state_aware --freeze_backbone_epochs 0 --exp_name "$EXP")

if [[ "${1:-}" == --worker ]]; then
    exec 7>"$WORKSPACE/baselines_compare/runs_joint/.runtime.lock"
    flock -n 7 || exit 75
    exec 8>"$ROOT/checkpoints/.strict_training_runtime.lock"
    flock -n 8 || exit 75
    export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    printf 'STARTED %s\n' "$(TZ=Asia/Shanghai date -Is)" > "$RUN/status.txt"
    printf '%q ' "$PYTHON" -u script/train_dino_corrected_v7.py "${ARGS[@]}" > "$RUN/command.txt"
    set +e
    "$PYTHON" -u script/train_dino_corrected_v7.py "${ARGS[@]}" 2>&1 | tee -a "$RUN/train.log"
    codes=("${PIPESTATUS[@]}");set -e
    printf 'FINISHED %s python_exit=%s tee_exit=%s\n' "$(TZ=Asia/Shanghai date -Is)" "${codes[0]}" "${codes[1]}" >> "$RUN/status.txt"
    [[ "${codes[1]}" -eq 0 ]] || exit "${codes[1]}"
    exit "${codes[0]}"
fi

if [[ -f "$RUN/status.txt" ]] || [[ -f "$RUN/latest.pth" ]]; then
    echo "ALREADY_STARTED $RUN; no restart";exit 0
fi
"$PYTHON" script/check_dino_ready.py
"$PYTHON" script/check_dino_handoff.py
if pgrep -af 'python.*(train_[a-zA-Z0-9_]+\.py|eval_[a-zA-Z0-9_]+\.py|resume_v11\.py)';then
    echo 'WAIT existing train/evaluation process';exit 75
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]];then
    echo 'WAIT GPU compute process';exit 75
fi
if [[ "${1:-}" == --check-only ]];then echo READY_FOR_CORRECTED_MAIN;exit 0;fi
exec 9>"$ROOT/checkpoints/.strict_training_launch.lock"
flock -n 9 || exit 75
exec 6>"$WORKSPACE/baselines_compare/runs_joint/.launch.lock"
flock -n 6 || exit 75
[[ ! -e "$RUN" ]] || { echo 'Existing partially prepared run; inspect it, do not overwrite/retry';exit 2; }
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON" script/verify_dino_integration.py
"$PYTHON" -u script/train_dino_corrected_v7.py "${ARGS[@]}" --prepare-run
"$PYTHON" -u script/train_dino_corrected_v7.py "${ARGS[@]}" --smoke-test 2>&1 | tee "$RUN/preflight_smoke.log"
screen -dmS "$SESSION" bash "$ROOT/launch_dino_corrected.sh" --worker
for attempt in {1..40};do
    [[ -f "$RUN/python.pid" ]] && break
    sleep .5
done
[[ -f "$RUN/python.pid" ]] || { echo 'Worker failed to initialize; inspect retained artifacts';exit 2; }
kill -0 "$(cat "$RUN/python.pid")"
echo "LAUNCHED $SESSION $RUN"
