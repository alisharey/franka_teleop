# BAKU policy: data upload, training (HPC) and deployment (this PC)

The BAKU code is the submodule `vla/BAKU` (alisharey/BAKU, branch
`franka-teleop`; its `FRANKA.md` explains the dataset, actions and
normalization). Training runs on the HPC in a checkout of the same branch
(`cluster.env`). This folder has the robot-PC side.

| Script | Does |
|---|---|
| `upload_episodes.sh --task "..." <episodes>` | exports episodes (robot state, TeleMoMa actions, color videos; no depth; `tools/baku/export.sh`) and uploads them to the HPC |
| `sync_baku.sh push` / `pull` | copy local edits of `vla/BAKU` to the HPC checkout (commit there) / update `vla/BAKU` to the branch's latest commit |
| `fetch_run.sh <run>/snapshot/<step>.pt` | copies a trained snapshot and its config from the HPC to `runs/` |
| `setup_local.sh` | once: checks out `vla/BAKU`, creates `.venv` (PyTorch 2.3.1 cu121, the cluster env's versions) |
| `deploy.sh <snapshot> [--record http://localhost:8765] [--safe-export DIR]` | runs the policy on the robot (`deploy.py`) |

## Running a policy on the robot

1. Terminal 1: `./record.sh --policy` (cameras, recorder, teleop listening for
   actions on UDP 47820).
2. Terminal 2: `vla/baku/deploy.sh vla/baku/runs/<run>/snapshot/50000.pt --record http://localhost:8765`.
   Enter starts a rollout, Enter stops it (then label it: s success, f failure,
   i failure with intervention, x discard); `h` = home pose; `q` = quit.

Each action becomes a target (current pose + action) that the teleop tracks
with the Quest's gains, speed limits, IK and workspace fence, as during the
demonstrations (`src/policy_input.h`). No action for `policy_timeout_ms`
(300 ms) stops the arm. Keep the stop button in reach. The GPU needs about
1 GB free.

Per rollout: `~/franka_policy_rollouts/<id>/rollout.npz` (actions sent,
VQ-BeT entropies, action-token features, policy input images); with
`--record` an episode in the recorder (cameras + robot bag, success label);
with `--safe-export` the failure-detection files
`task<T>--ep<E>--succ<0|1>.{csv,pkl,meta.json}`.
