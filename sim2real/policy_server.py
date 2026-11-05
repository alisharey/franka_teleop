#!/usr/bin/env python3
"""Serves a trained BAKU snapshot to the Isaac Lab evaluation (isaac/cup_demos.py --policy), which runs
in a different python environment. Run it in BAKU's env (it imports vla/BAKU/baku/franka/policy.py):

    python sim2real/policy_server.py <run>/snapshot/<step>.pt [--baku <BAKU checkout>] [--port 8790]
One policy copy per simulated environment (each keeps its own observation history).
Protocol (multiprocessing.connection, local TCP), npz both ways: the client sends 'env' (k,) environment
indices, 'reset' (k,) 1 = new episode for that environment, and the frames 'pixels' (static camera) and
'pixels_egocentric' (wrist camera), (k, H, W, 3) RGB uint8, already background-filtered; the server
answers 'action' (k, 7) TeleMoMa actions and 'rate_hz'. A request with only 'rate' returns 'rate_hz'.
"""
import argparse
import copy
import io
import sys
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np


def pack(**arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def unpack(data):
    return dict(np.load(io.BytesIO(data)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('snapshot')
    parser.add_argument('--baku', default=str(Path(__file__).resolve().parents[1] / 'vla' / 'BAKU'))
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--task', default='0')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.baku) / 'baku'))
    from franka.policy import FrankaPolicy
    first = FrankaPolicy(args.snapshot, task=int(args.task) if args.task.isdigit() else args.task)
    policies = {}
    print('policy %s, task %r, %.1f Hz; listening on 127.0.0.1:%d' % (args.snapshot, first.task_text, first.rate_hz, args.port),
          flush=True)
    with Listener(('127.0.0.1', args.port)) as listener:
        while True:
            with listener.accept() as connection:
                try:
                    while True:
                        request = unpack(connection.recv_bytes())
                        if 'env' not in request:
                            connection.send_bytes(pack(rate_hz=np.float64(first.rate_hz)))
                            continue
                        actions = []
                        for row, env in enumerate(request['env'].tolist()):
                            if env not in policies:
                                policies[env] = first if not policies else copy.deepcopy(first)
                            policy = policies[env]
                            if request['reset'][row]:
                                policy.reset()
                            actions.append(policy.act({k: request[k][row] for k in policy.pixel_keys}))
                        connection.send_bytes(pack(action=np.array(actions), rate_hz=np.float64(first.rate_hz)))
                except EOFError:
                    print('client disconnected', flush=True)


if __name__ == '__main__':
    main()
