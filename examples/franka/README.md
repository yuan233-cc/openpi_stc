# Fine-tuning pi0.5 on the Franka sponge dataset

The raw dataset is a LeRobot v3 recording, but its recorder-native state and
action fields are not directly suitable for OpenPI. The converter follows the
custom-data workflow in the repository README and writes a new LeRobot dataset
with these training features:

- `image`: the third-person RGB camera, stored as one MP4 per episode
- `state`: `[x, y, z, rx, ry, rz, gripper]`
- `actions`: `[x, y, z, rx, ry, rz, gripper]` absolute next poses
- `task`: `pick up the sponge`

The source quaternion is converted to an axis-angle rotation vector. The source
`action` pose is the next state and is stored as an absolute target. When the
data loader creates a 50-step action chunk, OpenPI subtracts that chunk's
initial observation state from all six pose dimensions. The gripper stays
absolute. During inference, OpenPI adds the initial state back to the predicted
pose deltas, so the policy returns absolute target poses. The two finger widths
are averaged and normalized to `[0, 1]` using the Franka finger travel of 0.04 m.

The converter streams the source Parquet in small batches and processes one
episode at a time. Camera video is decoded one frame at a time, so it does not
load all 42 episodes or all video frames into RAM. Image files are written
asynchronously by 10 threads by default. Episode commits remain sequential
because LeRobot uses one mutable episode buffer and shared dataset metadata.

## 1. Convert the data

From the OpenPI repository root, run:

```bash
uv run examples/franka/convert_franka_sponge_to_lerobot.py \
  --data-dir /home/prs/Yuan_Feng/data/raw_pick_up_sponge_42_action_next_state \
  --repo-id prs/franka_sponge_pi05
```

The result is written to `$HF_LEROBOT_HOME/prs/franka_sponge_pi05` (normally
`~/.cache/huggingface/lerobot/prs/franka_sponge_pi05`). Use `--overwrite` to
replace an earlier conversion. For a quick conversion test, add
`--max-frames 10 --output-root /tmp/franka_sponge_smoke`.

The default writer configuration is usually a good balance for one camera. To
experiment with process-based image writing, for example, use:

```bash
uv run examples/franka/convert_franka_sponge_to_lerobot.py \
  --image-writer-processes 2 \
  --image-writer-threads 4 \
  --overwrite
```

This parallelizes temporary image encoding/writing, not complete episodes.
Increasing these values can speed up conversion on machines with fast storage,
but also increases CPU and RAM usage.

## 2. Compute normalization statistics

The `pi05_franka_sponge` training config points to the converted repo id and
maps its fields through `FrankaInputs`.

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_sponge
```

## 3. Train pi0.5

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_sponge \
  --exp-name=pick_up_sponge \
  --overwrite
```

## Glass dataset with D455 and D405 cameras

The merged glass recording has two video features. Both participate in training:

- `observation.images.d455` becomes the `base_0_rgb` model view.
- `observation.images.d405` becomes the `left_wrist_0_rgb` model view.

The third pi0.5 image slot remains a masked black placeholder.

```bash
uv run examples/franka/convert_franka_sponge_to_lerobot.py \
  --data-dir /home/prs/Yuan_Feng/data/pick_up_glass_51ep_merged_cropped_20260827_action_next_state \
  --repo-id prs/franka_glass_pi05_two_camera \
  --output-root /home/prs/Yuan_Feng/data/franka_glass_pi05_two_camera \
  --image-key observation.images.d455 \
  --wrist-image-key observation.images.d405
```

The training loader resolves a repo id through `$HF_LEROBOT_HOME`. If the
dataset is kept at the custom output path above, link it into the default cache:

```bash
mkdir -p ~/.cache/huggingface/lerobot/prs
ln -s /home/prs/Yuan_Feng/data/franka_glass_pi05_two_camera \
  ~/.cache/huggingface/lerobot/prs/franka_glass_pi05_two_camera
```

Then compute statistics and train with the glass-specific full-parameter pi0.5
config. The 3.35B-parameter model does not fit full AdamW training on one 32 GB
GPU; use FSDP across multiple GPUs (for example, four 32 GB GPUs):

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_glass

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_glass \
  --exp-name=pick_up_glass_full \
  --fsdp-devices=4 \
  --no-wandb-enabled
```

Adjust `CUDA_VISIBLE_DEVICES` and `--fsdp-devices` together for the available
multi-GPU machine. The global batch size is 8 and must be divisible by the total
number of visible devices.

At inference time, the ROS websocket observation must likewise provide both
encoded image messages: `image` for D455 and `wrist_image` for D405. The server
maps them to the same two model slots used during training.
