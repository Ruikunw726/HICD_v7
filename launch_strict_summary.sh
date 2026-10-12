#!/usr/bin/env bash
# Presentation-only publisher, detached from Codex's terminal session.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN="$(realpath -- "${1:?Usage: launch_strict_summary.sh RUN_DIR}")"
EXP="${RUN##*/}"
PYTHON=/root/miniconda/envs/mamba/bin/python
[[ "$EXP" =~ ^[a-zA-Z0-9_]+$ ]] || { echo 'Invalid run name'; exit 1; }
[[ -d "$RUN/metrics" && -f "$RUN/python.pid" ]] || { echo 'Missing run artifacts/PID'; exit 1; }
TRAINING_PID="$(cat "$RUN/python.pid")"
[[ "$TRAINING_PID" =~ ^[0-9]+$ ]] || { echo 'Invalid training PID'; exit 1; }
training_cmd="$(tr '\0' ' ' < "/proc/$TRAINING_PID/cmdline")"
[[ "$training_cmd" == *train_strict_v7.py* && "$training_cmd" == *"$EXP"* ]] || {
    echo 'Exact experiment is not running; use report_strict_epochs.py without --watch for backfill'; exit 1;
}

publisher_alive() {
    [[ -f "$RUN/evaluation_reporter.pid" ]] || return 1
    local publisher_pid publisher_cmd
    publisher_pid="$(cat "$RUN/evaluation_reporter.pid")"
    [[ "$publisher_pid" =~ ^[0-9]+$ && -r "/proc/$publisher_pid/cmdline" ]] || return 1
    publisher_cmd="$(tr '\0' ' ' < "/proc/$publisher_pid/cmdline")"
    [[ "$publisher_cmd" == *report_strict_epochs.py* && "$publisher_cmd" == *"$EXP"* ]]
}

if publisher_alive; then
    echo "SUMMARY_ALREADY_RUNNING pid=$(cat "$RUN/evaluation_reporter.pid")"
    exit 0
fi
cd "$ROOT"
SESSION="hicd_summary_$EXP"
screen -L -Logfile "$RUN/evaluation_reporter.log" -dmS "$SESSION" \
    "$PYTHON" -u "$ROOT/script/report_strict_epochs.py" --run-dir "$RUN" \
    --watch --training-pid "$TRAINING_PID"
# Keep the launching WSL shell alive through screen/Python initialization.
# A successful fork alone is not evidence that a detached worker survived.
for attempt in {1..20}; do
    if publisher_alive && screen -ls | grep -F ".$SESSION" | grep -q 'Detached'; then
        echo "SUMMARY_RUNNING pid=$(cat "$RUN/evaluation_reporter.pid") session=$SESSION"
        exit 0
    fi
    sleep 0.5
done
echo "Summary publisher did not initialize; inspect $RUN/evaluation_reporter.log"
exit 1
