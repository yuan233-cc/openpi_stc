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

Then compute statistics and train with the glass-specific pi0.5 LoRA config. It
uses LoRA for both the 2B language backbone and the 300M action expert and fits
on one 32 GB GPU:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_franka_glass

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_glass \
  --exp-name=pick_up_glass_lora \
  --no-wandb-enabled
```

At inference time, the ROS websocket observation must likewise provide both
encoded image messages using the recorder keys: `observation.images.d455` and
`observation.images.d405`. The server maps them to the same two model slots used
during training. For compatibility, it also accepts `d455`/`d405`, nested
`images.d455`/`images.d405`, and the legacy `image`/`wrist_image` keys.

### Strict 101-episode glass dataset

This dataset stores continuous measured finger widths in `observation.state`
and explicit binary open/close labels in the action gripper dimension. Convert
it without changing those command labels:

```bash
uv run examples/franka/convert_franka_sponge_to_lerobot.py \
  --data-dir /home/prs/ros_ml_ws/src/franka_data_recorder/data/pick_up_glass_101ep_merged_cropped_fixed_action_raw_gripper_15hz_strict_20260828 \
  --repo-id prs/franka_glass_101ep_pi05_two_camera_15hz \
  --output-root /home/prs/Yuan_Feng/data/franka_glass_101ep_pi05_two_camera_15hz \
  --gripper-action-mode binary_command \
  --default-task "pick up the glass"
```

The training CLI accepts the converted directory directly; a Hugging Face cache
symlink is not required. This is also the recommended way to train after copying
or downloading the dataset onto another computer:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_glass_101ep \
  --data.dataset-root=/path/to/franka_glass_101ep_pi05_two_camera_15hz \
  --exp-name=pick_up_glass_101ep_full \
  --fsdp-devices=4
```

The local directory must be the dataset root containing `meta/`, `data/`, and
`videos/`. The `pi05_franka_glass_101ep` config loads its matching norm stats
through OpenPI's standard config asset directory,
`assets/pi05_franka_glass_101ep/prs/franka_glass_101ep_pi05_two_camera/norm_stats.json`.
The root `assets/` directory is tracked, so no machine-specific asset path or
additional norm-stat computation is needed after cloning the repository.
If `--data.dataset-root` is omitted, the original repo-id/cache lookup remains
available.

For example, if the dataset is copied to `/data/openpi_datasets` on the other
computer, use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_glass_101ep \
  --data.dataset-root=/data/openpi_datasets/franka_glass_101ep_pi05_two_camera_15hz \
  --exp-name=pick_up_glass_101ep_full \
  --fsdp-devices=4
```

This 15 Hz config updates all pi0.5 parameters and does not use LoRA or a
freeze filter. Full-parameter AdamW training usually needs multiple GPUs; set
`CUDA_VISIBLE_DEVICES` and `--fsdp-devices` to match the available machine. The
global batch size is 16 and must be divisible by the number of visible devices.

### Strict 101-episode glass dataset at 30 Hz

The full-rate dataset uses the same continuous measured gripper state and binary
gripper command convention. Its output repo id and directory explicitly include
`30hz`:

```bash
uv run examples/franka/convert_franka_sponge_to_lerobot.py \
  --data-dir /home/prs/ros_ml_ws/src/franka_data_recorder/data/pick_up_glass_101ep_merged_cropped_fixed_action_raw_gripper_20260828 \
  --repo-id prs/franka_glass_101ep_pi05_two_camera_30hz \
  --output-root /home/prs/Yuan_Feng/data/franka_glass_101ep_pi05_two_camera_30hz \
  --gripper-action-mode binary_command \
  --default-task "pick up the glass"
```

`pi05_franka_glass_101ep_30hz` explicitly reuses the same tracked norm stats, so
do not run `compute_norm_stats.py` again. Train with:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_glass_101ep_30hz \
  --data.dataset-root=/path/to/franka_glass_101ep_pi05_two_camera_30hz \
  --exp-name=pick_up_glass_101ep_30hz_lora
```
