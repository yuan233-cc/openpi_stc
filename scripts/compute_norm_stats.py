"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
from openpi.policies import franka_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def franka_action_chunks(states: np.ndarray, actions: np.ndarray, start: int, stop: int, horizon: int):
    """Match LeRobot's episode-end padding and the training relative-pose transform."""
    indices = np.minimum(np.arange(start, stop)[:, None] + np.arange(horizon), len(actions) - 1)
    relative = franka_policy.FrankaRelativeActions()
    return np.stack(
        [
            relative({"state": states[index], "actions": actions[targets]})["actions"]
            for index, targets in zip(range(start, stop), indices, strict=True)
        ]
    )


def compute_franka_lowdim_stats(data_config: _config.DataConfig, horizon: int):
    """Full-dataset Franka statistics without reading RGB or dropping a final batch.

    This path deliberately supports only the existing 7-D Franka transform contract.
    Other configurations must use the regular data loader.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    import pyarrow.parquet as pq

    if [type(t) for t in data_config.data_transforms.inputs] != [
        franka_policy.FrankaInputs,
        franka_policy.FrankaRelativeActions,
    ] or tuple(data_config.action_sequence_keys) != ("actions",):
        raise ValueError("--lowdim-only requires the 7-D FrankaInputs + FrankaRelativeActions pipeline")
    # Refuse custom repacking that could alter the pose/gripper semantics.
    repack = transforms.compose(data_config.repack_transforms.inputs)
    sentinel = np.arange(7, dtype=np.float32)
    packed = repack(
        {"state": sentinel, "actions": sentinel[None], "image": None, "wrist_image": None, "prompt": "check"}
    )
    if not np.array_equal(packed["observation/state"], sentinel) or not np.array_equal(
        packed["actions"], sentinel[None]
    ):
        raise ValueError("--lowdim-only requires direct state/actions repacking")
    root = pathlib.Path(data_config.dataset_root).expanduser().resolve() if data_config.dataset_root else None
    meta = LeRobotDatasetMetadata(data_config.repo_id, root=root)
    if tuple(meta.features["state"]["shape"]) != (7,) or tuple(meta.features["actions"]["shape"]) != (7,):
        raise ValueError("--lowdim-only requires 7-D state/actions")
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    frames_seen = 0
    for episode_index in tqdm.tqdm(sorted(meta.episodes), desc="Computing full Franka stats"):
        path = meta.root / meta.get_data_file_path(episode_index)
        table = pq.read_table(path, columns=["state", "actions", "frame_index", "episode_index", "timestamp"])
        states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        length = int(meta.episodes[episode_index]["length"])
        if states.shape != (length, 7) or actions.shape != (length, 7) or length == 0:
            raise ValueError(f"invalid episode shape/length: {episode_index}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"nonfinite values in episode {episode_index}")
        np.testing.assert_array_equal(table["frame_index"].to_numpy(), np.arange(length))
        np.testing.assert_array_equal(table["episode_index"].to_numpy(), np.full(length, episode_index))
        np.testing.assert_allclose(table["timestamp"].to_numpy(), np.arange(length) / meta.fps, atol=1e-4, rtol=0)
        for start in range(0, length, 256):
            stop = min(start + 256, length)
            stats["state"].update(states[start:stop])
            stats["actions"].update(franka_action_chunks(states, actions, start, stop, horizon))
        frames_seen += length
    if frames_seen != meta.total_frames:
        raise ValueError(f"frame count mismatch: {frames_seen} vs {meta.total_frames}")
    print(
        f"Used all {frames_seen} states and {frames_seen * horizon} action targets across {len(meta.episodes)} episodes"
    )
    return {key: value.get_statistics() for key, value in stats.items()}


def main(
    config_name: str, max_frames: int | None = None, dataset_root: str | None = None, *, lowdim_only: bool = False
):
    config = _config.get_config(config_name)
    if dataset_root is not None:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, dataset_root=dataset_root))
    data_config = config.data.create(config.assets_dirs, config.model)

    if lowdim_only:
        if max_frames is not None:
            raise ValueError("--lowdim-only computes the full dataset; omit --max-frames")
        norm_stats = compute_franka_lowdim_stats(data_config, config.model.action_horizon)
        output_path = config.assets_dirs / data_config.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
