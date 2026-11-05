#!/usr/bin/env python3
"""Compares action statistics of exported episodes, e.g. scripted sim demos vs real teleop demos.

    python3 sim2real/compare_actions.py --a <sim export root> --b <real export episodes...> [--plot out.png]
Per group: episode length, position action norm |target - current| (driven steps) percentiles,
end-effector speed percentiles, rotation action norm, gripper-closed fraction.
"""
import argparse
from pathlib import Path

import numpy as np


def episodes(paths):
    out = []
    for p in paths:
        p = Path(p)
        out += [p] if (p / 'lowdim.npz').exists() else sorted(q.parent for q in p.glob('*/lowdim.npz'))
    return out


def stats(paths):
    pos, rot, speed, lengths, closed = [], [], [], [], []
    for e in episodes(paths):
        d = np.load(e / 'lowdim.npz')
        a = d['action']
        driven = d['movement_enabled'].astype(bool) if 'movement_enabled' in d else np.ones(len(a), bool)
        pos.append(np.linalg.norm(a[driven, :3], axis=1))
        rot.append(np.linalg.norm(a[driven, 3:6], axis=1))
        dt = np.diff(d['t'])
        speed.append(np.linalg.norm(np.diff(d['ee_pos'], axis=0), axis=1) / np.maximum(dt, 1e-3))
        lengths.append(d['t'][-1] - d['t'][0])
        closed.append(float((a[:, 6] < 0.5).mean()))
    cat = lambda x: np.concatenate(x) if x else np.zeros(1)  # noqa: E731
    return {'episodes': len(lengths), 'length_s': np.array(lengths), 'pos': cat(pos), 'rot': cat(rot),
            'speed': cat(speed), 'closed': np.array(closed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', nargs='+', required=True)
    parser.add_argument('--b', nargs='+', required=True)
    parser.add_argument('--names', default='sim,real')
    parser.add_argument('--plot')
    args = parser.parse_args()
    names = args.names.split(',')
    groups = {names[0]: stats(args.a), names[1]: stats(args.b)}
    q = [10, 50, 90, 99]
    for name, s in groups.items():
        print('%s: %d episodes, length %.1f s median (%.1f-%.1f), gripper closed %.0f%% of steps' % (
            name, s['episodes'], np.median(s['length_s']), s['length_s'].min(), s['length_s'].max(), 100 * np.mean(s['closed'])))
        print('  |pos action| mm  p%s: %s' % ('/'.join(map(str, q)), np.round(np.percentile(s['pos'], q) * 1000, 1)))
        print('  |rot action| deg p%s: %s' % ('/'.join(map(str, q)), np.round(np.degrees(np.percentile(s['rot'], q)), 2)))
        print('  ee speed mm/s    p%s: %s' % ('/'.join(map(str, q)), np.round(np.percentile(s['speed'], q) * 1000, 0)))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
        for name, s in groups.items():
            axes[0].hist(s['pos'] * 1000, bins=60, range=(0, 80), density=True, alpha=0.5, label=name)
            axes[1].hist(s['speed'] * 1000, bins=60, range=(0, 300), density=True, alpha=0.5, label=name)
            axes[2].hist(np.degrees(s['rot']), bins=60, range=(0, 10), density=True, alpha=0.5, label=name)
        for ax, title in zip(axes, ('|position action| [mm]', 'end-effector speed [mm/s]', '|rotation action| [deg]')):
            ax.set_title(title)
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
        print('plot', args.plot)


if __name__ == '__main__':
    main()
