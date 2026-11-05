# BAKU policy: data upload, training (HPC) and deployment (this PC)

The BAKU code is the submodule `vla/BAKU` (alisharey/BAKU, branch
`franka-teleop`; its `FRANKA.md` explains the dataset, actions and
normalization). Training runs on the HPC in a checkout of the same branch
(`cluster.env`). This folder has the robot-PC side.

| Script | Does |
|---|---|
| `upload_episodes.sh --task "..." <episodes>` | exports episodes (robot state, TeleMoMa actions, color videos; no depth; `tools/baku/export.sh`) and uploads them to the HPC |
| `sync_baku.sh push` / `pull` | copy local edits of `vla/BAKU` to the HPC checkout (commit there) / update `vla/BAKU` to the branch's latest commit |
| `fetch_run.sh <run>/snapshot/<step>.pt` | copies a trained snapshot and its config from the HPC to `runs/` (the page lists everything in `runs/`) |
| `setup_local.sh` | once: checks out `vla/BAKU`, creates `.venv` (PyTorch 2.3.1 cu121, the cluster env's versions) |
| `deploy.py` | the policy server (started by `./record.sh --deploy`); `deploy.sh --no-motion` runs it alone, preview only |

## Running a policy on the robot

```bash
./record.sh --deploy
```

starts the cameras, the recorder, the policy server and the teleop listening
for policy actions. On the web page (http://localhost:8765) the **Policy**
panel:

1. **Checkpoint** -> **Load** (a second or two).
2. **Preview (no motion)**: the policy runs on the live cameras and shows what
   it would do (move/turn per step, gripper, the next second of its plan,
   VQ-BeT entropy). Nothing is sent to the robot.
3. **Go home**: the arm moves to `home_joints` (config/teleop.conf); the chip
   turns to "at home".
4. **Run episode** (with "home first" and "record episode" ticked by default):
   the arm follows the policy until **Stop** or 60 s (600 steps).
5. Label it: **Success**, **Failure**, **Failure, I intervened**, or **Discard**.

Each action becomes a target (current pose + action) that the teleop tracks
with the Quest's gains, speed limits, IK and workspace fence, as during the
demonstrations (`src/policy_input.h`). No action for `policy_timeout_ms`
(300 ms) stops the arm; a camera without frames for 1 s, or the teleop
stopping, ends the episode. Keep the stop button in reach.

## What a rollout records

- `~/franka_policy_rollouts/<time>_<checkpoint>_task<N>/rollout.npz`, per policy
  step: the action sent and the raw policy output, the two 128x128 images the
  policy saw, the 10-step action chunk it predicted, the picked VQ-BeT codes, the
  full code distributions (2 x 16), their entropies and the picked codes'
  probabilities, the 256-d action-token feature, inference time and camera
  frame age; `rollout.json`: checkpoint, task, steps, outcome, recorder episode.
- With "record episode": a recorder episode named `policy_<checkpoint>_<time>`
  (camera videos, the 1 kHz robot bag including `/teleop/quest` with the
  executed target and tracking error, success label, notes with the checkpoint).
- With `--safe-export DIR` (deploy.py): the failure-detection files
  `task<T>--ep<E>--succ<0|1>.{csv,pkl,meta.json}`, as interpretability/monitor.py.
