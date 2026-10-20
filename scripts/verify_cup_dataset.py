"""Verify the public cup ZIP contents and exercise the real pi05 training input pipeline."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import tyro

from openpi.training import config as training_config
from openpi.training import data_loader


def main(dataset_root: str, *, check_hashes: bool = True):
    root = Path(dataset_root).expanduser().resolve()
    manifest = json.loads((root / "release_manifest.json").read_text())
    info = json.loads((root / "meta/info.json").read_text())
    if not (root / "VALIDATED_COMPLETE").is_file():
        raise ValueError("Missing complete-dataset marker")
    if (info["total_episodes"], info["total_frames"], info["fps"]) != (98, 44529, 10):
        raise ValueError("Unexpected dataset size or frequency")
    if len(list(root.glob("videos/**/*.mp4"))) != 196 or len(list(root.glob("data/**/*.parquet"))) != 98:
        raise ValueError("Missing episode videos/parquet files")
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing/invalid file: {entry['path']}")
        if check_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"SHA256 mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} files; 98 episodes / 44,529 frames / 196 videos")
    config = training_config.get_config("pi05_franka_cup_10hz")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, dataset_root=str(root)),
        assets_base_dir=str(Path(__file__).resolve().parents[1] / "assets"),
        batch_size=2,
        num_workers=0,
    )
    loader = data_loader.create_data_loader(config, num_batches=1, shuffle=False)
    if loader.data_config().norm_stats is None:
        raise ValueError("Cup normalization statistics were not loaded")
    observation, actions = next(iter(loader))
    if tuple(actions.shape) != (2, 50, 32) or tuple(observation.state.shape) != (2, 32):
        raise ValueError(f"Unexpected training shapes: {actions.shape}, {observation.state.shape}")
    if not np.isfinite(np.asarray(actions)).all() or not np.isfinite(np.asarray(observation.state)).all():
        raise ValueError("Nonfinite normalized inputs")
    for key in ("base_0_rgb", "left_wrist_0_rgb"):
        if tuple(observation.images[key].shape) != (2, 224, 224, 3):
            raise ValueError(f"Unexpected RGB shape for {key}")
        if not np.asarray(observation.image_masks[key]).all():
            raise ValueError(f"Missing camera: {key}")
    if np.asarray(observation.image_masks["right_wrist_0_rgb"]).any():
        raise ValueError("Unexpected third camera")
    print("PASS: actual pi05 input pipeline, cup norm stats, RGB decode/resize, tokenizer, state and action shapes")
    print("No model training or model-weight allocation performed; smoke-test batch size 2 only (training stays 64).")


if __name__ == "__main__":
    tyro.cli(main)
