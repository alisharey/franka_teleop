# Click to move

Click a point in a camera view and the gripper goes there, around an obstacle you select (tracked with
EdgeTAM). Two versions, both started with the cameras and the teleop by their script (close `./record.sh`
first; Ctrl-C ends everything):

| | Page | Planner |
|---|---|---|
| `click_to_move.sh` | http://localhost:8780, both cameras | own: around or over the obstacle's keep-out (gripper only) |
| `curobo_move.sh` | http://localhost:8781, static camera | cuRobo v0.9 (whole arm) against the table you identify and the obstacle |

They need `config/camera_calibration.json` (`tools/calibration/calibration_server.sh`). `curobo_move.sh` keeps
the table you identify in `config/table_plane.json`. EdgeTAM's weights (56 MB) are downloaded once to
`data/models/hf`.

## Setup

cuRobo needs torch >= 2.5, so it has its own environment (about 8 GB, in `tools/click_to_move/curobo/`,
not committed). EdgeTAM is installed into it too:

```bash
cd tools/click_to_move
git clone --depth 1 --branch v0.9.0 https://github.com/NVlabs/curobo.git curobo
cd curobo
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install ".[cu12]" opencv-python-headless
.venv/bin/pip install "transformers==4.57.6" "huggingface_hub>=0.34,<1.0" "tokenizers>=0.22,<=0.23.0" timm safetensors regex
```

`click_to_move.sh` runs in `tools/click_to_move/.venv`: a venv with `include-system-site-packages = true`
whose torch comes from the BAKU environment (a `.pth` file pointing at its site-packages), plus
`transformers==4.57.6`, `pillow>=11`, `timm` (`--no-deps`) and `huggingface_hub`.
