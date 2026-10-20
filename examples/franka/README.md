# Fine-tuning pi0.5 on the Franka sponge dataset

For ROS WebSocket inference commands and the checkpoint/config selection table,
see [README_inference.md](README_inference.md).

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

To see the OpenPI action curves without printing action values in the terminal,
start `scripts/serve_ros_ws_policy.py` with `--plot-action-values`. The server
renders the seven policy dimensions (`x`, `y`, `z`, `rx`, `ry`, `rz`, and
`gripper`) over the complete chunk horizon and updates a browser dashboard after
every inference. Plotting runs in a background worker and drops stale plot jobs
rather than delaying robot inference.

The option is disabled by default. For example:

```bash
uv run python scripts/serve_ros_ws_policy.py \
  --mode POLICY \
  --action-format ABSOLUTE_ROTVEC \
  --plot-action-values \
  --action-plot-host 127.0.0.1 \
  --action-plot-port 8766 \
  --host 0.0.0.0 \
  --port 8765 \
  policy:checkpoint \
  --policy.config=<config-name> \
  --policy.dir=<checkpoint-path>
```

Open <http://127.0.0.1:8766> in a browser. If OpenPI runs on another machine,
forward the dashboard port first:

```bash
ssh -L 8766:127.0.0.1:8766 user@openpi-server
```

Then open the same local browser URL. The latest PNG and HTML page are also
stored in `/tmp/openpi_action_curves` by default; change this with
`--action-plot-dir`.

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

This 15 Hz config updates all pi0.5 parameters, does not use LoRA or a freeze
filter, and maintains an EMA copy with decay `0.99`. It uses a global batch size
of 64, 1,000 warmup steps, and cosine decay from `2.5e-5` to `2.5e-6` over
15,000 steps. Full-parameter AdamW plus EMA requires high-memory GPUs; set
`CUDA_VISIBLE_DEVICES` and `--fsdp-devices` to match the available machine. The
global batch size must be divisible by the number of visible devices.

### Put two small glasses, strict 15 Hz

The `pi05_put_two_smallglass_15hz_full` config performs full-parameter pi0.5
fine-tuning with EMA (`0.99`) on the 117-episode, two-camera dataset. Its repo id
is `yuan1119/put_two_smallglass_15hz`, and its matching quantile normalization
statistics are tracked in the config's standard assets directory. After cloning
this repository, no `--data.repo-id` or `--data.assets.*` overrides are needed.

Download `data.zip` from the dataset repository and extract it so that the
selected dataset root directly contains `meta/`, `data/`, and `videos/`. The
config defaults target two FSDP GPUs with a global batch of 128 (64 per GPU), so
run:

```bash
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_put_two_smallglass_15hz_full \
  --data.dataset-root=/path/to/put_two_samllglass_117ep_15hz \
  --exp-name=put_two_samllglass_117ep_15hz_full
```

Compared with the previous global-batch-64 recipe, the two-GPU profile uses a
global batch of 128 (64 per GPU). Its peak learning rate is linearly scaled to
`5e-5`, with a 500-step warmup and 15,000 total training steps. Checkpoints are
saved every 5,000 steps and at the final step. This is a memory-intensive
full-parameter configuration; reduce the batch only if the target GPUs run out
of memory.

Run ROS WebSocket inference with the downloaded 3,000-step checkpoint:

```bash
cd /home/prs/Yuan_Feng/workspace/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/serve_ros_ws_policy.py \
  --mode POLICY \
  --default-prompt "pick up the glass and put it into the container" \
  --input-format LIBERO_SINGLE_IMAGE \
  --action-format ABSOLUTE_ROTVEC \
  --host 0.0.0.0 \
  --port 8765 \
  policy:checkpoint \
  --policy.config=pi05_put_two_smallglass_15hz_full \
  --policy.dir=/home/prs/Yuan_Feng/workspace/openpi/checkpoints/pick_two_small_3000_50
```

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

## Green cup, D435if + D405, 10 Hz

`pi05_franka_cup_10hz` uses the complete 98-episode, 44,529-frame dataset
`yuan1119/franka_cup_d435if_d405_pi05_10hz`. `image` is D435if RGB at 1920x1080;
`wrist_image` is D405 RGB center-cropped to 640x480. Each recorded task supplies
its own language prompt. States are absolute 7-D poses/openings; stored actions
are absolute next measured poses/openings. The training transform converts pose
actions to deltas relative to the chunk's initial state using SO(3) rotations;
the measured gripper opening remains absolute. A 50-step chunk spans 5 seconds.

The default is full-parameter training: `gemma_2b` + `gemma_300m`, with no
LoRA adapters or frozen model parameters. It follows the old global-batch-64
`pi05_put_two_smallglass_15hz_full` recipe (before commit `857b98e`):

| Setting | Value |
| --- | --- |
| Global batch size | 64 (divided across visible GPUs) |
| Data workers | 4 |
| Training steps | 15,000 |
| LR schedule | Linear warmup + cosine decay |
| Warmup steps | 1,000 |
| Peak / final learning rate | `2.5e-5` / `2.5e-6` |
| Decay steps | 15,000, including warmup |
| Optimizer | AdamW, beta1 `0.9`, beta2 `0.95`, epsilon `1e-8` |
| Weight decay / gradient norm clip | `1e-10` / `1.0` |
| EMA | `0.99` |
| Save interval / periodic retention | 1,000 / 3,000 steps |
| Log interval / seed | 100 steps / 42 |
| FSDP devices | 1 by default; override for the cluster |
| Model compute dtype | `bfloat16` |
| Action horizon | 50 steps at 10 Hz (5 seconds) |

The later two-glass configuration's global batch 128 means 64 per GPU on two
GPUs; this cup configuration uses **global batch 64**. Adjust FSDP to the cluster
hardware, for example `--fsdp-devices=2` for a two-GPU FSDP group. The global
batch must be divisible by the number of visible GPUs, and the visible GPU
count must be divisible by the FSDP group size. Existing cup norm stats remain
valid because the data, relative-pose transform and action horizon are unchanged.
No machine-specific data path is stored in this configuration.
From the repository root, set one cluster-local path:

```bash
export DATASET_DIR=/path/to/franka_cup_d435if_d405_pi05_10hz_v21

# Recompute only when changing the dataset, pose transforms, or action horizon.
uv run scripts/compute_norm_stats.py \
  --config-name pi05_franka_cup_10hz \
  --dataset-root "$DATASET_DIR" \
  --lowdim-only

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_cup_10hz \
  --data.dataset-root "$DATASET_DIR" \
  --exp-name green_cup_10hz \
  --no-wandb-enabled
```

The cup-specific statistics live at
`assets/pi05_franka_cup_10hz/yuan1119/franka_cup_d435if_d405_pi05_10hz/norm_stats.json`.
Copy that asset directory together with the updated repository code to the
cluster; the dataset itself can stay in any directory. There is no need to copy
local Hugging Face cache symlinks or repeat data conversion. The cluster also
needs access to the configured pi05 base weights and tokenizer cache/download.

`--lowdim-only` is restricted to this 7-D Franka transform contract. It reads
Parquet state/actions without decoding images, uses the same relative-pose
transform as training, and includes every frame and all 50 action targets per
start frame, including LeRobot's episode-end padding. Thus a final short batch
is not dropped when computing statistics. It generates mean/std and the 1%/99%
quantiles used by pi05. The standard image-decoding statistics path is unchanged.

### Cup 数据集：集群训练交接说明

请在集群执行以下流程，不要重新转换、重新裁剪或再次把 next state 移位为 action。
这是 **pi05 全量训练，不是 LoRA**。使用单节点 GPU；本仓库训练脚本不支持多节点。
全量训练需要较大显存，建议先分配单张 80GB GPU；多 GPU 可用 FSDP 分片，实际
batch=64 的显存是否够用需由集群实测，不要静默降低 batch 或改成 LoRA。

- 代码：<https://github.com/yuan233-cc/openpi_stc>，分支 `main`。
- 公共数据：<https://huggingface.co/datasets/yuan1119/franka_cup_d435if_d405_pi05_10hz>。
- ZIP：`franka_cup_d435if_d405_pi05_10hz_v21.zip`，ZIP64 格式。
- 配置：`pi05_franka_cup_10hz`，上表超参数不变。
- 归一化统计已随 Git 提交，无需重算。数据仓库是 ZIP 发布格式，必须解压并指定
  `--data.dataset-root`，不能指望 LeRobot 根据 repo ID 自动下载并解压 ZIP。
- ZIP 是传输包，训练必须读取可随机访问的解压目录；若平台支持挂载 ZIP，必须提供
  可正常随机 seek 的 MP4 文件，否则先解压到持久盘。下载包加解压至少预留 15GB，
  模型、依赖和训练 checkpoint 另需空间；不要把 checkpoint 写到易清理的临时盘。

下列 `/path/to/...` 由集群操作方换成实际挂载路径，不依赖原采集电脑的目录。

```bash
# 1. 同步代码；已有仓库时先检查 git status，不要覆盖本地改动。
git clone --recurse-submodules https://github.com/yuan233-cc/openpi_stc.git
cd openpi_stc
git submodule update --init --recursive
git rev-parse HEAD  # 保存本次使用的提交 ID

# 2. 使用仓库锁定的依赖；需要 uv、NVIDIA 驱动和 FFmpeg 运行库。
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# 3. 下载公共 ZIP（不需要 HF token），并验证 SHA256。
export CUP_DOWNLOAD_DIR=/path/to/persistent/downloads/cup
export CUP_DATA_PARENT=/path/to/persistent/datasets
mkdir -p "$CUP_DOWNLOAD_DIR" "$CUP_DATA_PARENT"
uv run python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id="yuan1119/franka_cup_d435if_d405_pi05_10hz", repo_type="dataset", local_dir=os.environ["CUP_DOWNLOAD_DIR"])'
(cd "$CUP_DOWNLOAD_DIR" && sha256sum -c SHA256SUMS)

# 若平台已提供 ZIP 挂载/下载，跳过下载，仍需检查校验和与完整标记。
# 不要解压覆盖已有数据目录；若目录已存在，先验证，再决定是否复用。
unzip -n "$CUP_DOWNLOAD_DIR/franka_cup_d435if_d405_pi05_10hz_v21.zip" -d "$CUP_DATA_PARENT"
export DATASET_DIR="$CUP_DATA_PARENT/franka_cup_d435if_d405_pi05_10hz_v21"

# 4. 完整文件哈希检查 + 真正训练输入流水线检查（不启动模型训练）。
# 首次可能下载 tokenizer，需要联网或预置缓存。
JAX_PLATFORMS=cpu uv run scripts/verify_cup_dataset.py --dataset-root "$DATASET_DIR"

# 5. 在已分配 GPU 的作业内确认 JAX 看得到 GPU。
uv run python -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'

# 6. 单节点训练。单 GPU 用 1；双 GPU 可改为 2，四 GPU 可改为 4。
# CUDA_VISIBLE_DEVICES 应由调度器管理；global batch 始终是 64。
export CUP_FSDP_DEVICES=1
export CUP_CHECKPOINT_DIR=/path/to/persistent/checkpoints
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_franka_cup_10hz \
  --data.dataset-root "$DATASET_DIR" \
  --exp-name cup_full_b64_run1 \
  --checkpoint-base-dir "$CUP_CHECKPOINT_DIR" \
  --fsdp-devices "$CUP_FSDP_DEVICES" \
  --no-wandb-enabled
```

checkpoint 将写入 `$CUP_CHECKPOINT_DIR/pi05_franka_cup_10hz/cup_full_b64_run1/`。
恢复同一个实验时，在训练命令追加 `--resume`；不要加 `--overwrite`。新实验使用新
`--exp-name`。FSDP 组大小必须整除可见 GPU 数，global batch 必须能被可见 GPU 数整除。
上述命令只能在实际分配到 GPU 的进程/作业内启动，具体 Slurm/PBS 参数由集群决定。

首次训练还需要下载 `gs://openpi-assets/checkpoints/pi05_base/params`、tokenizer
及相关缓存；公共 cup ZIP **不包含模型权重**。离线计算节点需预置缓存（参照
`OPENPI_DATA_HOME` / `HF_HOME` 的实现与集群存储策略），或先在联网节点准备。
不要全局设置 `JAX_PLATFORMS=cpu`，它仅用于上面的数据验证命令。

已知采集限制：原 recorder 按周期取各话题最新值，不是严格曝光/机器人状态时间同步；
语言描述来自下达指令，不保证与视觉估计的最终物体位移完全一致。打包保留完整
98 段，不自动筛选或改标签。相关审计报告随公共发布提供，不应将“转换校验通过”
等同于“所有原始监督无噪声”。训练启动前请记录 Git SHA、数据 ZIP SHA256、GPU
数量/型号和实际命令；不要重复生成不同语义的 action 或使用其他任务的 norm stats。
