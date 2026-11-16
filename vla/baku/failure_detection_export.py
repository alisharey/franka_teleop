#!/usr/bin/env python3
"""Converts labelled policy rollouts (deploy.py's data/rollouts/<run>/rollout.{json,npz})
into the Failure_Detection shared-dataset BAKU format; vla/baku/upload_rollouts.sh runs it into a
temporary folder and uploads the result to the HPC store.

Per labelled run (the Policy panel's label in rollout.json: success, failure or intervention; unlabelled and
discarded runs are skipped):
    task<T>--ep<E>--succ<0|1>.csv        per policy step: the policy's actions (action/dx..dgripper), VQ-BeT
                                         token entropies and probabilities (2 native codebooks, padded to 7)
    task<T>--ep<E>--succ<0|1>.pkl        hidden_states (T, 1, 256) float32 = the action token before the
                                         VQ-BeT head; task_suite_name, task_id, task_description,
                                         episode_idx, episode_success
    task<T>--ep<E>--succ<0|1>.meta.json  provenance, outcome, intervention, length, end reason
    task<T>--ep<E>--succ<0|1>.npz        what was executed: sent_action (after the deploy clip), raw_action,
                                         code distributions, action chunks, timing
T: stable per task text (tasks.json, kept with the uploaded data). E: the run's start time as
YYMMDDHHMMSS, so the same run always gets the same file names.

    python3 failure_detection_export.py --out DIR [--rollouts data/rollouts] [--tasks tasks.json]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

ACTION_NAMES = ['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'dgripper']
OUTCOMES = ('success', 'failure', 'intervention')


def demo_dataset(info):
    """Training-data provenance: recorded at save time; for older runs, from the checkpoint's run config."""
    if info.get('demo_dataset'):
        return info['demo_dataset']
    config = Path(info['checkpoint']['path']).parents[1] / '.hydra' / 'config.yaml'
    if config.exists():
        for line in config.read_text().splitlines():
            if line.strip().startswith('path:') and 'expert_demos' in line:
                return {'path': line.split(':', 1)[1].strip()}
    return None


def export_run(folder, out, tasks, suite, defaults):
    info = json.loads((folder / 'rollout.json').read_text())
    if info.get('outcome') not in OUTCOMES or not info.get('steps'):
        return None
    log = np.load(folder / 'rollout.npz')
    task_text = info['task']
    if task_text not in tasks:
        tasks[task_text] = max(tasks.values(), default=-1) + 1
    task_id = tasks[task_text]
    episode = int(folder.name[2:8] + folder.name[9:15])  # 20261108_172213_... -> 261108172213
    success = info['outcome'] == 'success'
    stem = out / ('task%d--ep%d--succ%d' % (task_id, episode, 1 if success else 0))
    raw, sent = log['raw_action'], log['action']
    header = (['action/%s' % n for n in ACTION_NAMES] + ['action/token_%d_entropy' % i for i in range(7)] +
              ['action/token_%d_prob' % i for i in range(7)])
    rows = np.concatenate([raw, log['token_entropy'], log['token_prob']], axis=1)
    with open(str(stem) + '.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join('%.9g' % v for v in row) + '\n')
    import torch
    with open(str(stem) + '.pkl', 'wb') as f:
        pickle.dump({'hidden_states': torch.tensor(log['feature'], dtype=torch.float32).unsqueeze(1),
                     'task_suite_name': suite, 'task_id': task_id, 'task_description': task_text,
                     'episode_idx': episode, 'episode_success': 1 if success else 0}, f)
    clipped = int(np.sum(np.any(np.abs(sent - raw) > 1e-12, axis=1)))
    np.savez_compressed(str(stem) + '.npz', t=log['t'], raw_action=raw, sent_action=sent,
                        code_probs=log['code_probs'], codes=log['codes'], action_chunk=log['action_chunk'],
                        compute_ms=log['compute_ms'], camera_age_s=log['camera_age_s'])
    clip = info.get('clip', {})
    meta = {'policy': 'baku', 'suite': suite, 'kind': 'natural', 'task_id': task_id, 'episode_idx': episode,
            'success': 1 if success else 0, 'intervention': info['outcome'] == 'intervention',
            'task_description': task_text, 'checkpoint': info['checkpoint']['path'],
            'demo_dataset_version': demo_dataset(info),
            'robot_id': info.get('robot_id', defaults['robot_id']),
            'camera_calibration_version': info.get('camera_calibration_version', defaults['camera_calibration']),
            'control_hz': info['rate_hz'],
            'action_filter': 'deploy clip |pos| <= %s m, |rot| <= %s rad; gamepad_teleop input=policy: target '
                             'tracking (quest gains), speed caps, workspace fence, contact limit, IK' % (
                                 clip.get('position_m'), clip.get('rotation_rad')),
            'executed_action': 'the CSV holds the policy actions; the .npz sent_action holds what was sent after '
                               'the clip (clipped_steps counts the steps it changed)',
            'steps': int(info['steps']), 'duration_s': float(log['t'][-1]), 'ended': info.get('ended'),
            'late_ticks': info.get('late_ticks'), 'clipped_steps': clipped,
            'rollout_id': folder.name,
            'recording': '%s/recording' % folder.name if (folder / 'recording').is_dir() else None}
    Path(str(stem) + '.meta.json').write_text(json.dumps(meta, indent=2, sort_keys=True) + '\n')
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--rollouts', type=Path, default=Path(__file__).resolve().parents[2] / 'data' / 'rollouts')
    parser.add_argument('--tasks', type=Path, help='existing tasks.json (task text -> id) to extend')
    parser.add_argument('--suite', default='real_franka_v1')
    parser.add_argument('--robot-id', default='franka_fr3')
    parser.add_argument('--camera-calibration', default='unversioned', help='for runs saved before it was recorded')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(args.tasks.read_text()) if args.tasks and args.tasks.exists() else {}
    defaults = {'robot_id': args.robot_id, 'camera_calibration': args.camera_calibration}
    counts = {}
    for folder in sorted(p.parent for p in args.rollouts.glob('*/rollout.json')):
        stem = export_run(folder, args.out, tasks, args.suite, defaults)
        if stem:
            outcome = 'success' if stem.name.endswith('succ1') else json.loads(
                (stem.parent / (stem.name + '.meta.json')).read_text())['intervention'] and 'intervention' or 'failure'
            counts[outcome] = counts.get(outcome, 0) + 1
    (args.out / 'tasks.json').write_text(json.dumps(tasks, indent=1, sort_keys=True) + '\n')
    print('exported %s; tasks %s' % (counts or 'nothing (no labelled runs)', tasks))


if __name__ == '__main__':
    main()
