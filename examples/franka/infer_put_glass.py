"""Run one local inference step with the 15 Hz Franka glass checkpoint.

The policy expects two RGB images and the current 7-D Franka state:

    [x, y, z, rx, ry, rz, gripper]

Position is in metres, rotation is an axis-angle rotation vector in radians,
and gripper is normalized to [0, 1] (0=closed, 1=open). The returned 50-step
action chunk uses the same ordering and contains absolute target poses.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/put_glass_101ep_15hz_full"
DEFAULT_CONFIG = "pi05_franka_glass_101ep"
ACTION_NAMES = ("x", "y", "z", "rx", "ry", "rz", "gripper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d455-image",
        type=Path,
        required=True,
        help="Path to the current D455 RGB image.",
    )
    parser.add_argument(
        "--d405-image",
        type=Path,
        required=True,
        help="Path to the current D405 wrist-camera RGB image.",
    )
    parser.add_argument(
        "--state",
        type=float,
        nargs=7,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ", "GRIPPER"),
        help="Current state: xyz [m], rotation vector [rad], normalized gripper [0,1].",
    )
    parser.add_argument(
        "--prompt",
        default="pick up the glass",
        help="Language instruction supplied to the policy.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"OpenPI training config name (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of diffusion sampling steps (default: 10).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional output path. Without this flag, JSON is printed to stdout.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def validate_inputs(args: argparse.Namespace) -> tuple[Path, np.ndarray]:
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params directory does not exist: {checkpoint / 'params'}")
    if not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(f"Checkpoint assets directory does not exist: {checkpoint / 'assets'}")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive")

    state = np.asarray(args.state, dtype=np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError("--state must contain only finite values")
    if not 0.0 <= state[6] <= 1.0:
        raise ValueError("The normalized gripper state must be in [0, 1]")
    return checkpoint, state


def make_json_payload(
    *,
    checkpoint: Path,
    config_name: str,
    prompt: str,
    actions: np.ndarray,
    policy_timing: dict[str, Any],
) -> dict[str, Any]:
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != len(ACTION_NAMES):
        raise ValueError(f"Expected actions with shape [horizon, 7], got {actions.shape}")

    # This dataset uses binary open/close targets, but diffusion inference may
    # produce small overshoots. Keep commands inside the robot's valid range.
    actions = actions.astype(np.float32, copy=True)
    actions[:, 6] = np.clip(actions[:, 6], 0.0, 1.0)
    return {
        "checkpoint": str(checkpoint),
        "config": config_name,
        "prompt": prompt,
        "action_names": list(ACTION_NAMES),
        "action_format": "absolute_xyz_rotation_vector_gripper",
        "control_hz": 15,
        "actions": actions.tolist(),
        "policy_timing": policy_timing,
    }


def main() -> None:
    args = parse_args()
    checkpoint, state = validate_inputs(args)
    observation = {
        "observation/image": load_rgb(args.d455_image),
        "observation/wrist_image": load_rgb(args.d405_image),
        "observation/state": state,
        "prompt": args.prompt,
    }

    logging.info("Loading config %s", args.config)
    train_config = _config.get_config(args.config)
    logging.info("Loading checkpoint %s", checkpoint)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=args.prompt,
        sample_kwargs={"num_steps": args.num_steps},
    )

    result = policy.infer(observation)
    payload = make_json_payload(
        checkpoint=checkpoint,
        config_name=args.config,
        prompt=args.prompt,
        actions=np.asarray(result["actions"]),
        policy_timing=result.get("policy_timing", {}),
    )
    output = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.output_json is None:
        print(output)
    else:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{output}\n", encoding="utf-8")
        logging.info("Wrote %d actions to %s", len(payload["actions"]), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
