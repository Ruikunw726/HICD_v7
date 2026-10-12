#!/usr/bin/env bash
# Resume only: preserve existing sessions, experiment definition and diagnostics.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
PYTHON=/root/miniconda/envs/mamba/bin/python
CKPT_DIR="$PROJECT_DIR/checkpoints/v11_no_t4"
SESSION=hicd_v11_resume

if [[ "${1:-}" == --worker ]]; then
    AUDIT="$2"
    CHECKPOINT="$3"
    exec 8>"$CKPT_DIR/.resume_runtime.lock"
    flock -n 8 || { echo 'Existing resume worker owns runtime lock'; exit 75; }
    export PYTHONPATH="$PROJECT_DIR" PYTHONUNBUFFERED=1
    printf 'STARTED %s\n' "$(date -Is)" > "$AUDIT/status.txt"
    printf '%q ' "$PYTHON" -u script/resume_v11.py --mode run --checkpoint "$CHECKPOINT" --audit-dir "$AUDIT" > "$AUDIT/command.txt"
    printf '\n' >> "$AUDIT/command.txt"
    set +e
    "$PYTHON" -u script/resume_v11.py --mode run --checkpoint "$CHECKPOINT" --audit-dir "$AUDIT" 2>&1 | tee -a "$CKPT_DIR/train.log" "$AUDIT/train_resume.log"
    codes=("${PIPESTATUS[@]}")
    set -e
    printf 'FINISHED %s python_exit=%s tee_exit=%s\n' "$(date -Is)" "${codes[0]}" "${codes[1]}" >> "$AUDIT/status.txt"
    [[ "${codes[1]}" -eq 0 ]] || exit "${codes[1]}"
    exit "${codes[0]}"
fi

[[ -d "$CKPT_DIR" && -x "$PYTHON" ]] || { echo 'Missing experiment/Python'; exit 1; }
exec 9>"$CKPT_DIR/.resume_launch.lock"
flock -n 9 || { echo 'Another launcher is active'; exit 75; }
if screen -ls | grep -Eq 'hicd_v11([_.[:space:]]|$)' || pgrep -af 'python.*(train_full_v7\.py|resume_v11\.py.*--mode run)'; then
    echo 'Training/session already exists; refusing duplicate launch without stopping it.'
    exit 75
fi
CHECKPOINT="${1:-}"
if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'checkpoint_epoch_[0-9]*.pth' | sort -V | tail -n 1)"
fi
[[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]] || { echo 'Explicit resume checkpoint missing'; exit 1; }
CHECKPOINT="$(realpath -- "$CHECKPOINT")"
AUDIT="$CKPT_DIR/resume_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
"$PYTHON" -u script/resume_v11.py --mode verify --checkpoint "$CHECKPOINT" --audit-dir "$AUDIT"
screen -dmS "$SESSION" bash "$PROJECT_DIR/launch_v11_resume.sh" --worker "$AUDIT" "$CHECKPOINT"
# Wait for the child to acquire its runtime lock before the WSL launcher returns.
# A socket alone can be stale and is not evidence of a running worker.
for attempt in {1..20}; do
    [[ -f "$AUDIT/status.txt" ]] && break
    sleep 0.5
done
[[ -f "$AUDIT/status.txt" ]] || { echo 'Screen worker failed to initialize; inspect the audit directory'; exit 1; }
screen -ls | grep -Eq "\\.${SESSION}[[:space:]].*Detached" || { echo 'Screen session did not remain detached'; exit 1; }
printf 'SESSION=%s\nAUDIT=%s\nCHECKPOINT=%s\n' "$SESSION" "$AUDIT" "$CHECKPOINT"
