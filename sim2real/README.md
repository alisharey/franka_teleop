# sim2real: cup stacking from sim-only demos

Real episodes -> digital twin of our table -> scripted demos in Isaac Lab (HPC)
-> BAKU trained on sim only -> deployed on the FR3. The gap is bridged by the
**background-filtered observation** (`observation.py`): every pixel's depth
becomes a point in the robot base frame and only what lies in a box above our
table (table, cups, robot) is kept, the rest is black. The same function runs
on the sim training data and on the live camera frames.

| Path | Does |
|---|---|
| `observation.py` | `BackgroundFilter.apply(camera, color, depth_mm, ee_pose)`; static camera cam_2: leveled pose + depth scale from the scene; wrist camera cam_1: O_T_EE x hand-eye (`config/camera_calibration.json`) |
| `scene/export_replay.py` | one episode's robot motion (q, gripper, O_T_EE) per camera frame, from the bag (system Python + ROS) |
| `scene/estimate_scene.py` | table plane/edges, cups (color, size, positions), camera check from the first frames of the episodes -> `scene.json` |
| `scene/refine_static_camera.py` | static camera depth scale from where the gripper grasps/releases the cups (FK) -> `scene.json` |
| `scene/table_texture.py` | top-down table texture from the static camera (objects removed by a median) |
| `scenes/<name>/` | the finished twin used downstream: `scene.json`, `table_texture.png/.json` |
| `isaac/cup_demos.py` | scripted stacking demos with physics in `--num-envs` parallel scenes (BAKU export format + depth), and `--policy` evaluation of a trained run |
| `isaac/replica.py` | kinematic replay of a real episode rendered next to the real cameras (twin check) |
| `isaac/run.sh` | runs either on an HPC compute node (`SCRIPT=cup_demos.py` default, or `replica.py`) |
| `isaac/mirror_usd.py` | local copy of Isaac Lab's Panda USD (`--robot-usd`; the asset server is flaky from the nodes) |
| `filter_episode.py` | applies the filter to a real episode (`--real`, check), a sim episode (`--sim`) or a whole sim export (`--sim-export`, training data) |

Camera calibration (`config/camera_calibration.json`: intrinsics, T_base_cam_2,
T_ee_cam_cam_1) is general infrastructure in `tools/calibration/`.

## Pipeline

```bash
PY=tools/calibration/.venv/bin/python   # cv2 + numpy (created by tools/calibration/*.sh)
W=~/franka_scene/cups_20261024          # work folder

# 1. calibrate the cameras -> config/camera_calibration.json
tools/calibration/calibration_server.sh          # or: tools/calibration/calibrate.sh <episode>

# 2. record real demos (./record.sh), then estimate the scene
$PY sim2real/scene/estimate_scene.py --out $W ~/franka_teleop_data/20261024_*

# 3. static camera depth scale (needs the robot motion of each episode)
source /opt/ros/humble/setup.bash && source install/setup.bash
for e in ~/franka_teleop_data/20261024_*; do
  python3 sim2real/scene/export_replay.py $e $W/replays/episode_${e##*_}_replay.npz; done
python3 sim2real/scene/refine_static_camera.py --scene $W/scene.json --replays $W/replays --picked blue --target red
$PY sim2real/scene/estimate_scene.py --depth-scale <printed> --out $W ~/franka_teleop_data/20261024_*  # rerun

# 4. table texture, then publish the twin
$PY sim2real/scene/table_texture.py --scene $W/scene.json --out $W/table_texture.png ~/franka_teleop_data/20261024_*
mkdir -p sim2real/scenes/cups_20261024 && cp $W/scene.json $W/table_texture.{png,json} sim2real/scenes/cups_20261024/

# check: filter a real episode (look at the two mp4s)
$PY sim2real/filter_episode.py --real ~/franka_teleop_data/<episode> --replay $W/replays/episode_<n>_replay.npz --out /tmp/filtered
```

On the HPC (a `--data` folder = `scenes/<name>/` + `config/camera_calibration.json`
copied as `calibration.json`; `--robot-usd` = the mirrored Panda, below):

```bash
# 5. sim demos (allocated node; run.sh sets up Isaac Lab)
srun --overlap --jobid=<job> /usr/bin/env OMNI_KIT_ACCEPT_EULA=YES sim2real/isaac/run.sh \
    --data <data> --robot-usd <assets>/panda/panda_instanceable.usd --episodes 200 --num-envs 32 --seed <n> --out <raw>
# (~6 min for 100 demos with 32 environments on one RTX 5000 Ada; failed grasps/stacks are dropped)
# twin check against a real episode: SCRIPT=replica.py ... --data <data with replays + real mp4s> --episode 0081 --out <dir>

# 6. filter -> training episodes (episode.json gets "observation": "background_filter")
python sim2real/filter_episode.py --sim-export <raw> --out <filtered>

# 7. BAKU dataset + training (vla/BAKU/FRANKA.md), and mark the dataset as filtered
python franka/build_dataset.py <filtered>/* --out ../expert_demos/<name> --subsample 3
echo '{"type": "background_filter", "scene": "sim2real/scenes/cups_20261024/scene.json"}' > ../expert_demos/<name>/observation.json
python train_franka.py dataloader.bc_dataset.path=$PWD/../expert_demos/<name> init_from=<LIBERO snapshot>

# 8. evaluate in Isaac before the robot: random cup positions, the same control as deployment
python sim2real/policy_server.py <run>/snapshot/<step>.pt --baku <BAKU checkout> --port 8791 &   # env baku
srun ... sim2real/isaac/run.sh --data <data> --robot-usd <usd> --policy 127.0.0.1:8791 --episodes 50 \
    --num-envs 32 --seed 900 --out <eval>   # -> <eval>/eval.json (success rate) + eval_<n>_*.mp4 (policy inputs)
```

## SAC in Isaac Lab (skrl)

`isaac/sac_cups.py` (started by `cup_demos.py --sac`) trains SAC on the same scene: the actor sees the
filtered static + wrist images (84x84) + end-effector position, gripper width and command; the critic also
gets the cup poses (training only). 200 scripted demos go into a separate buffer (half of every batch,
plus a behavior-cloning term). Actions: 10 Hz target offsets (5 cm x [dx dy dz]) through the teleop
tracking law, open/close; the gripper keeps its orientation. Needs `pip install skrl` in the Isaac Lab env.

```bash
srun ... sim2real/isaac/run.sh --data <data> --robot-usd <usd> --sac --num-envs 32 --sac-steps 6000 --out <out>
#   -> <out>/policy_final.pt, eval_final.json (+ videos), skrl checkpoints in <out>/sac/checkpoints
#   continue a run: --sac-init <out>/sac/checkpoints/agent_<n>.pt ; evaluate a policy: --sac-eval <policy.pt>
```

First result (run sac_v2): 43/50 random Isaac episodes (BAKU VQ-BeT on the same demos: 16/50).
Deployable as a run folder `vla/baku/runs/<name>/` with `snapshot/<name>.pt` (the actor's state_dict),
`policy.json` `{"type": "sac", "task": "..."}` and `observation.json`; `deploy.py` then uses
`sim2real/sac_policy.py` (torch only) instead of BAKU.

## Deploying a sim2real run

`observation.json` travels with the run: `vla/baku/fetch_run.sh` copies it from
the run folder, else from the training dataset folder, into
`vla/baku/runs/<run>/`. When the Policy panel loads a checkpoint of such a run,
`deploy.py` also subscribes to `/cameras/<cam>/aligned_depth_to_color/image_raw`,
pairs each color frame with its depth frame (same stamp), takes O_T_EE from
`/teleop/state` at the frame's stamp for the wrist camera, and feeds
`BackgroundFilter(..., sim=False).apply(...)` output to the policy's usual
preprocessing (so `rollout.npz` logs the filtered 128x128 images). The panel shows
"observation: background filter" and the filter time next to the inference time.
Missing depth or robot pose counts like a stale camera (hold, end the episode
after 1 s). A run without the file is deployed unchanged. To mark a run by hand:

```bash
echo '{"type": "background_filter", "scene": "sim2real/scenes/cups_20261024/scene.json"}' \
    > vla/baku/runs/<run>/observation.json   # optional: "calibration", "box" [x0,x1,y0,y1,z0,z1]
```

Paths are relative to the repository. The filter costs ~20 ms per step for both
cameras at 640x480 (10 Hz policy). Then `./record.sh --deploy` as usual
(vla/baku/README.md). Dry run without the robot (sends nothing; isolated ROS domain):

```bash
export ROS_DOMAIN_ID=77
python3 tests/sim/fake_cameras.py ~/franka_teleop_data/<cup episode> --depth --state &   # ROS + install sourced
vla/baku/deploy.sh --no-motion --record "" --http-port 8799   # then POST /api/policy/load + /preview
```

## Panda USD for a local Isaac install

`sim2real/assets/` is git-ignored (3.3 MB from NVIDIA's asset server). Fetch it
once with Isaac Lab's Python, from the IsaacLab folder (URL for Isaac Lab 2.3 /
Isaac Sim 5.1; other versions: `FRANKA_PANDA_HIGH_PD_CFG.spawn.usd_path`):

```bash
./isaaclab.sh -p <repo>/sim2real/isaac/mirror_usd.py \
    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd \
    <repo>/sim2real/assets/panda
```
