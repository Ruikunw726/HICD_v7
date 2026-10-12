#!/usr/bin/env bash
# A new experiment, never v11 resume. All process protection is non-destructive.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON=/root/miniconda/envs/mamba/bin/python
ARGS=(--dataset 0617final_t123_disjoint_20261009 --config_dir datasets/configs --data_dir /mnt/d/CD/0617final_256_v2_disjoint_20261009
      --batch_size 32 --img_size 256 --max_epochs 100 --learning_rate 1e-4 --use_amp
      --num_workers 4 --backbone_type dinov3 --state_aware --freeze_backbone_epochs 0)

if [[ "${1:-}" == --worker ]]; then
    EXP="$2"
    RUN="$ROOT/checkpoints/$EXP"
    exec 8>"$ROOT/checkpoints/.strict_training_runtime.lock"
    flock -n 8 || { echo 'Another strict training worker owns GPU run lock'; exit 75; }
    export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
    printf 'STARTED %s\n' "$(TZ=Asia/Shanghai date -Is)" > "$RUN/status.txt"
    printf '%q ' "$PYTHON" -u script/train_strict_v7.py "${ARGS[@]}" --exp_name "$EXP" > "$RUN/command.txt"
    printf '\n' >> "$RUN/command.txt"
    set +e
    "$PYTHON" -u script/train_strict_v7.py "${ARGS[@]}" --exp_name "$EXP" 2>&1 | tee -a "$RUN/train.log"
    codes=("${PIPESTATUS[@]}")
    set -e
    printf 'FINISHED %s python_exit=%s tee_exit=%s\n' "$(TZ=Asia/Shanghai date -Is)" "${codes[0]}" "${codes[1]}" >> "$RUN/status.txt"
    [[ "${codes[1]}" -eq 0 ]] || exit "${codes[1]}"
    exit "${codes[0]}"
fi

EXP="${1:-v12_strict_joint_t123_disjoint_e100_20261009}"
[[ "$EXP" =~ ^v12_strict_joint_[a-zA-Z0-9_]+$ ]] || { echo 'Invalid experiment name'; exit 1; }
RUN="$ROOT/checkpoints/$EXP"
SESSION="hicd_$EXP"
exec 9>"$ROOT/checkpoints/.strict_training_launch.lock"
flock -n 9 || { echo 'Another strict launcher is running'; exit 75; }
if screen -ls | grep -E 'hicd_' || pgrep -af 'python.*(train_[a-zA-Z0-9_]+\.py|resume_v11\.py)'; then
    echo 'Existing HICD training/session detected; refusing duplicate launch without stopping it.'
    exit 75
fi
[[ ! -e "$RUN" ]] || { echo "Existing experiment directory: $RUN; choose a unique name"; exit 1; }
# Run the accepted core and adapter numerical tests before any GPU preflight.
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v
"$PYTHON" -u script/train_strict_v7.py "${ARGS[@]}" --exp_name "$EXP" --prepare-run
"$PYTHON" -u script/train_strict_v7.py "${ARGS[@]}" --exp_name "$EXP" --smoke-test 2>&1 | tee "$RUN/preflight_smoke.log"
screen -dmS "$SESSION" bash "$ROOT/launch_v12_strict.sh" --worker "$EXP"
for attempt in {1..20}; do
    [[ -f "$RUN/status.txt" ]] && break
    sleep 0.5
done
[[ -f "$RUN/status.txt" ]] || { echo 'Screen worker failed to initialize'; exit 1; }
screen -ls | grep -Eq "\\.${SESSION}[[:space:]].*Detached" || { echo 'Screen did not remain detached'; exit 1; }
printf 'EXPERIMENT=%s\nSESSION=%s\nRUN=%s\n' "$EXP" "$SESSION" "$RUN"
