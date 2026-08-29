"""Convert a recorder-native Franka LeRobot dataset into an OpenPI pi0.5 dataset.

The source dataset is already LeRobot v3, but its fields are recorder-native:

  observation.state = [x,y,z,qx,qy,qz,qw, wrench(6), gripper_0, gripper_1]
  action            = [x,y,z,qx,qy,qz,qw, gripper]
  observation.images.<camera> = video

This script writes a new LeRobot dataset with the field names and dimensions used
by OpenPI training:

  image   = third-person RGB image
  state   = [x,y,z,rx,ry,rz,gripper]                 absolute proprio
  actions = [x,y,z,rx,ry,rz,gripper]                  absolute next pose

Raw rotations are quaternions and are converted to axis-angle. The source
``action`` pose is an absolute next state, which the converter verifies and
stores without converting it to a delta. The gripper action can either be the
next measured width used by older datasets or an explicit binary open/close
command. During training, OpenPI constructs an action chunk and expresses every
pose relative to the chunk's initial observation:

  delta_position = target_position - current_position
  delta_rotation = Log(inverse(R_current) @ R_target)

The gripper remains absolute. During inference, OpenPI composes the initial pose
with the predicted pose deltas, returning absolute target poses.

Usage:
  uv run examples/franka/convert_franka_sponge_to_lerobot.py \
      --data-dir /home/prs/Yuan_Feng/data/raw_pick_up_sponge_42_action_next_state \
      --repo-id prs/franka_sponge_pi05

When ``--image-key`` is omitted, the converter prefers
``observation.images.base`` and then ``observation.images.d455``. If
``observation.images.d405`` is also present, it is automatically written as the
second ``wrist_image`` feature. Explicit keys can override either selection.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Literal

import einops
import imageio.v3 as iio
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch
import tyro

GRIPPER_MAX_FINGER_WIDTH = 0.04
DEFAULT_DATA_DIR = "/home/prs/Yuan_Feng/data/raw_pick_up_sponge_42_action_next_state"


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_image(image) -> np.ndarray:
    image = _to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _gripper_from_raw_state(state: np.ndarray) -> float:
    # The recorder stores two finger positions in meters.  OpenPI uses a single
    # normalized command/state value where 1=open and 0=closed.
    return float(np.clip(np.mean(state[13:15]) / GRIPPER_MAX_FINGER_WIDTH, 0.0, 1.0))


def _state_to_openpi(raw_state: np.ndarray) -> np.ndarray:
    raw_state = np.asarray(raw_state, dtype=np.float32)
    if raw_state.shape != (15,):
        raise ValueError(f"Expected a 15-D observation.state, got {raw_state.shape}")
    pos = raw_state[:3]
    rotvec = Rotation.from_quat(raw_state[3:7]).as_rotvec().astype(np.float32)
    gripper = np.array([_gripper_from_raw_state(raw_state)], dtype=np.float32)
    return np.concatenate([pos, rotvec, gripper]).astype(np.float32)


def _absolute_action_from_raw_action(raw_action: np.ndarray) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    if raw_action.shape != (8,):
        raise ValueError(f"Expected an 8-D action, got {raw_action.shape}")
    pos = raw_action[:3]
    rotvec = Rotation.from_quat(raw_action[3:7]).as_rotvec().astype(np.float32)
    gripper = np.array([float(np.clip(raw_action[7], 0.0, 1.0))], dtype=np.float32)
    return np.concatenate([pos, rotvec, gripper]).astype(np.float32)


def _task_map(data_dir: Path) -> dict[int, str]:
    tasks_path = data_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    tasks = pd.read_parquet(tasks_path)
    if "task_index" not in tasks.columns:
        return {}
    if "task" in tasks.columns:
        return {int(row.task_index): str(row.task) for row in tasks.itertuples(index=False)}
    if tasks.index.name == "task":
        return {int(row.task_index): str(task) for task, row in tasks.iterrows()}
    return {}


def _iter_lowdim_episodes(data_dir: Path, *, batch_size: int = 512, default_task: str = "do something"):
    """Stream source Parquet and yield one episode at a time."""
    task_by_index = _task_map(data_dir)
    data_files = sorted((data_dir / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {data_dir / 'data'}")

    episode_frames = []
    episode_index = None
    columns = ["observation.state", "action", "episode_index", "frame_index", "task_index"]
    for parquet_path in data_files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                row_episode = int(row["episode_index"])
                if episode_index is None:
                    episode_index = row_episode
                elif row_episode != episode_index:
                    if row_episode < episode_index:
                        raise ValueError("Source rows must be ordered by episode_index and frame_index")
                    yield episode_frames
                    episode_frames = []
                    episode_index = row_episode

                frame_index = int(row["frame_index"])
                if frame_index != len(episode_frames):
                    raise ValueError(
                        f"Episode {episode_index} has non-contiguous frame_index {frame_index}; "
                        f"expected {len(episode_frames)}"
                    )
                episode_frames.append(
                    {
                        "episode": row_episode,
                        "frame_index": frame_index,
                        "state": np.asarray(row["observation.state"], dtype=np.float32),
                        "action": np.asarray(row["action"], dtype=np.float32),
                        "task": task_by_index.get(int(row["task_index"]), default_task),
                    }
                )

    if episode_frames:
        yield episode_frames


def _validate_next_state_actions(
    frames: list[dict],
    *,
    gripper_action_mode: Literal["next_state", "binary_command"] = "next_state",
    atol: float = 1e-6,
) -> None:
    """Check that the pose action is the next state and validate its gripper label."""
    for index, frame in enumerate(frames):
        is_terminal = index + 1 == len(frames) or frame["episode"] != frames[index + 1]["episode"]
        target_state = frame["state"] if is_terminal else frames[index + 1]["state"]
        action = np.asarray(frame["action"], dtype=np.float32)
        if action.shape != (8,):
            raise ValueError(f"Episode {frame['episode']} frame {index} has action shape {action.shape}; expected (8,)")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Episode {frame['episode']} frame {index} has non-finite action values")

        if not np.allclose(action[:7], target_state[:7], rtol=0.0, atol=atol):
            max_error = float(np.max(np.abs(action[:7] - target_state[:7])))
            raise ValueError(
                f"Episode {frame['episode']} frame {index} pose action is not the next state "
                f"(max error {max_error:.3g}); this converter only supports action_next_state poses."
            )

        if gripper_action_mode == "next_state":
            expected_gripper = _gripper_from_raw_state(target_state)
            if not np.isclose(action[7], expected_gripper, rtol=0.0, atol=atol):
                raise ValueError(
                    f"Episode {frame['episode']} frame {index} gripper action {action[7]:.6g} "
                    f"does not match next-state gripper {expected_gripper:.6g}; use "
                    "--gripper-action-mode binary_command only for datasets with explicit 0/1 commands."
                )
        elif not (np.isclose(action[7], 0.0, rtol=0.0, atol=atol) or np.isclose(action[7], 1.0, rtol=0.0, atol=atol)):
            raise ValueError(
                f"Episode {frame['episode']} frame {index} binary gripper command must be 0 or 1, got {action[7]:.6g}"
            )


def _dataset_info(data_dir: Path) -> dict:
    with (data_dir / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_key(info: dict, requested_key: str | None) -> str:
    features = info.get("features", {})
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]

    if requested_key is not None:
        if requested_key not in video_keys:
            raise ValueError(f"Image key {requested_key!r} is not a video feature; available video keys: {video_keys}")
        return requested_key

    for preferred_key in ("observation.images.base", "observation.images.d455"):
        if preferred_key in video_keys:
            return preferred_key
    if len(video_keys) == 1:
        return video_keys[0]
    if not video_keys:
        raise ValueError("Source dataset has no video features")
    raise ValueError(f"Multiple video features found; select one with --image-key: {video_keys}")


def _resolve_wrist_image_key(info: dict, primary_key: str, requested_key: str | None) -> str | None:
    features = info.get("features", {})
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]

    if requested_key is not None:
        if requested_key not in video_keys:
            raise ValueError(
                f"Wrist image key {requested_key!r} is not a video feature; available video keys: {video_keys}"
            )
        if requested_key == primary_key:
            raise ValueError("Primary and wrist image keys must be different")
        return requested_key

    if "observation.images.d405" in video_keys and primary_key != "observation.images.d405":
        return "observation.images.d405"
    return None


def _iter_video_frames(data_dir: Path, image_key: str):
    video_root = data_dir / "videos" / image_key
    video_files = sorted(video_root.glob("chunk-*/file-*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No video files found under {video_root}")
    for video_path in video_files:
        print(f"  reading video {video_path.relative_to(data_dir)}")
        yield from iio.imiter(video_path, plugin="pyav")


def main(
    data_dir: str = DEFAULT_DATA_DIR,
    repo_id: str = "prs/franka_sponge_pi05",
    *,
    output_root: str | None = None,
    image_key: str | None = None,
    wrist_image_key: str | None = None,
    gripper_action_mode: Literal["next_state", "binary_command"] = "next_state",
    default_task: str = "pick up the sponge",
    image_writer_processes: int = 0,
    image_writer_threads: int = 10,
    max_frames: int | None = None,
    push_to_hub: bool = False,
    overwrite: bool = False,
) -> None:
    """Convert a recorder-native Franka dataset.

    Args:
        data_dir: Source LeRobot dataset directory.
        repo_id: Output LeRobot repo id.  OpenPI configs refer to this id.
        output_root: Optional output root. Defaults to HF_LEROBOT_HOME / repo_id.
        image_key: Source camera key. If omitted, select a known main-camera key automatically.
        wrist_image_key: Optional second camera key. D405 is selected automatically when available.
        gripper_action_mode: ``next_state`` verifies the action gripper against the next measured state;
            ``binary_command`` preserves and validates explicit 0/1 open/close command labels.
        default_task: Prompt used only when the source task metadata has no matching task.
        image_writer_processes: Worker processes used to write temporary image frames.
        image_writer_threads: Image-writer threads per process, or total threads when processes is zero.
        max_frames: Optional smoke-test limit.  If set, conversion stops after
            this many frames and saves the current partial episode.
        push_to_hub: Push the converted dataset to Hugging Face Hub.
        overwrite: Delete an existing output dataset before conversion.
    """
    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {data_path}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if image_writer_processes < 0 or image_writer_threads < 0:
        raise ValueError("Image-writer process and thread counts cannot be negative")
    if image_writer_processes == 0 and image_writer_threads == 0:
        raise ValueError("At least one image-writer process or thread is required")
    if image_writer_processes > 0 and image_writer_threads == 0:
        raise ValueError("--image-writer-threads must be positive when using writer processes")
    output_path = HF_LEROBOT_HOME / repo_id if output_root is None else Path(output_root).expanduser().resolve()

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_path}. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    info = _dataset_info(data_path)
    image_key = _resolve_image_key(info, image_key)
    wrist_image_key = _resolve_wrist_image_key(info, image_key, wrist_image_key)
    height, width, channels = info["features"][image_key]["shape"]
    features = {
        "image": {
            # Explicitly request video-backed storage. In LeRobot v2.1,
            # use_videos=True does not change a manually declared "image"
            # feature into a "video" feature.
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
    }
    if wrist_image_key is not None:
        wrist_height, wrist_width, wrist_channels = info["features"][wrist_image_key]["shape"]
        features["wrist_image"] = {
            "dtype": "video",
            "shape": (wrist_height, wrist_width, wrist_channels),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="franka_fr3",
        fps=int(info["fps"]),
        features=features,
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )
    expected_video_keys = {"image"}
    if wrist_image_key is not None:
        expected_video_keys.add("wrist_image")
    if set(dataset.meta.video_keys) != expected_video_keys:
        raise RuntimeError(f"Expected video features {expected_video_keys}, got video keys {dataset.meta.video_keys}")

    print(f"source: {data_path} ({info['total_episodes']} episodes, {info['total_frames']} frames @ {info['fps']} fps)")
    print(f"output: {output_path}")
    print(f"image key: {image_key} ({width}x{height}x{channels})")
    if wrist_image_key is not None:
        print(f"wrist image key: {wrist_image_key} ({wrist_width}x{wrist_height}x{wrist_channels})")
    print(
        f"actions=absolute recorded next-state pose; gripper={gripper_action_mode}; quaternion rotations -> axis-angle"
    )
    print(f"image writers={image_writer_processes} processes x {image_writer_threads} threads")

    expected_frames = int(info["total_frames"])
    expected_episodes = int(info["total_episodes"])
    target_frames = expected_frames if max_frames is None else min(max_frames, expected_frames)
    video_frames = {"image": _iter_video_frames(data_path, image_key)}
    if wrist_image_key is not None:
        video_frames["wrist_image"] = _iter_video_frames(data_path, wrist_image_key)
    seen_frames = 0
    seen_episodes = 0
    written_frames = 0
    for episode in _iter_lowdim_episodes(data_path, default_task=default_task):
        _validate_next_state_actions(episode, gripper_action_mode=gripper_action_mode)
        seen_frames += len(episode)
        seen_episodes += 1

        remaining = target_frames - written_frames
        if remaining <= 0:
            break
        frames_to_write = episode[:remaining]
        for frame in frames_to_write:
            source_images = {}
            for output_key, frame_iterator in video_frames.items():
                try:
                    source_images[output_key] = _parse_image(next(frame_iterator))
                except StopIteration as exc:
                    raise RuntimeError(
                        f"{output_key} video ended after {written_frames} frames, but {target_frames} are required"
                    ) from exc
            dataset.add_frame(
                {
                    **source_images,
                    "state": _state_to_openpi(frame["state"]),
                    "actions": _absolute_action_from_raw_action(frame["action"]),
                    "task": frame["task"],
                }
            )
            written_frames += 1

        dataset.save_episode()
        print(
            f"  wrote episode {episode[0]['episode']} "
            f"({len(frames_to_write)} frames, total {written_frames}/{target_frames})"
        )
        if written_frames >= target_frames:
            break

    if written_frames != target_frames:
        raise RuntimeError(f"Frame count mismatch: wrote={written_frames}, expected={target_frames}")
    if max_frames is None and (seen_frames != expected_frames or seen_episodes != expected_episodes):
        raise RuntimeError(
            f"Source count mismatch: parquet={seen_episodes} episodes/{seen_frames} frames, "
            f"info.json={expected_episodes} episodes/{expected_frames} frames"
        )

    if hasattr(dataset, "finalize"):
        dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub(tags=["franka", "pi05"], private=False, push_videos=True, license="apache-2.0")

    print(f"done -> {output_path}")
    print(f"Use this repo id in OpenPI config: {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)
