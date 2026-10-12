"""Read completed epoch artifacts and publish compact four-decimal summaries.

Read-only with respect to training inputs/metrics/checkpoints. A separate
summary log lets an already running Python process retain its loaded code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from strict_joint_reporting import format_epoch_summary


def append_completed_epochs(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    previous = output.read_text(encoding='utf-8') if output.exists() else ''
    seen = {int(epoch) for epoch in re.findall(r'^Epoch (\d+) \|', previous, flags=re.MULTILINE)}
    added = []
    for path in sorted((run_dir / 'metrics').glob('epoch_*.json')):
        match = re.fullmatch(r'epoch_(\d+)\.json', path.name)
        if match and int(match.group(1)) in seen:
            continue
        # Training writes artifacts by atomic rename; incomplete .tmp ignored.
        artifact = json.loads(path.read_text(encoding='utf-8'))
        if artifact['epoch'] in seen:
            continue
        text = format_epoch_summary(artifact)
        with output.open('a', encoding='utf-8') as stream:
            stream.write(text + '\n\n')
            stream.flush()
        seen.add(artifact['epoch'])
        added.append(artifact['epoch'])
        print(text, flush=True)
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--training-pid', type=int,
                        help='With --watch, stop when this exact training process exits')
    options = parser.parse_args()
    run_dir = Path(options.run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f'Experiment missing: {run_dir}')
    if options.watch and not options.training_pid:
        raise ValueError('--watch requires --training-pid')
    if options.watch:
        import fcntl
        # Only one live publisher per run; holding this descriptor holds lock.
        lock = (run_dir / '.evaluation_summary.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run_dir / 'evaluation_reporter.pid').write_text(str(os.getpid()) + '\n', encoding='utf-8')
    output = run_dir / 'evaluation_summary.log'
    while True:
        append_completed_epochs(run_dir, output)
        if not options.watch:
            return
        process = Path('/proc') / str(options.training_pid)
        try:
            command = (process / 'cmdline').read_bytes().replace(b'\x00', b' ')
        except FileNotFoundError:
            command = b''
        if b'train_strict_v7.py' not in command or run_dir.name.encode() not in command:
            # Read once more in case the final report arrived during this pass.
            append_completed_epochs(run_dir, output)
            return
        time.sleep(2)


if __name__ == '__main__':
    main()
