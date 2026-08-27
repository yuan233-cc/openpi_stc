"""Serve the ROS WebSocket robot-control protocol.

This server is the counterpart to the ROS 2 `robot_ws_client` package. It accepts
JSON observations with a base64-encoded ROS Image and returns the complete
policy action chunk as absolute target end-effector poses.

The default mode is `hold`, which simply returns the current pose as the target
pose. Use it first to test networking and image decoding before enabling policy
inference.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import enum
import json
import logging
import math
from typing import Any

import numpy as np
from openpi_client import image_tools
import tyro
import websockets.asyncio.server as _server

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    HOLD = "hold"
    POLICY = "policy"


class PolicyInputFormat(enum.Enum):
    LIBERO_SINGLE_IMAGE = "libero_single_image"
    RAW_MODEL = "raw_model"


class ActionFormat(enum.Enum):
    ABSOLUTE_POSE = "absolute_pose"
    ABSOLUTE_ROTVEC = "absolute_rotvec"
    DELTA_ROTVEC = "delta_rotvec"


@dataclasses.dataclass
class Checkpoint:
    """Load an OpenPI policy checkpoint."""

    # Training config name, for example `pi05_libero`.
    config: str
    # Checkpoint directory, for example `gs://openpi-assets/checkpoints/pi05_libero`.
    dir: str


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8765
    mode: Mode = Mode.HOLD

    # Used when mode=policy.
    policy: Checkpoint | None = None
    default_prompt: str = "do something"
    input_format: PolicyInputFormat = PolicyInputFormat.LIBERO_SINGLE_IMAGE
    action_format: ActionFormat = ActionFormat.ABSOLUTE_ROTVEC
    # ROS image preprocessing before policy inference.
    resize_size: int = 224
    # One Franka finger joint is approximately 0.04 m when fully open. The
    # training dataset stores mean(finger_joint_positions) / this value.
    gripper_max_finger_width: float = 0.04

    # If true, any policy/inference error returns a hold action instead of closing the websocket.
    hold_on_error: bool = True


def decode_ros_image(image_msg: dict[str, Any]) -> np.ndarray:
    """Decode a JSON ROS Image payload into RGB uint8 HWC."""
    raw = base64.b64decode(image_msg["data_b64"])
    height = int(image_msg["height"])
    width = int(image_msg["width"])
    step = int(image_msg["step"])
    encoding = str(image_msg["encoding"])

    if encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"Unsupported image encoding: {encoding}")

    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, step))
    img = arr[:, : width * 3].reshape((height, width, 3))
    if encoding == "bgr8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def quat_xyzw_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a rotation vector without scipy."""
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    x, y, z, w = quat / norm
    w = float(np.clip(w, -1.0, 1.0))
    sin_half = math.sqrt(max(0.0, 1.0 - w * w))
    angle = 2.0 * math.atan2(sin_half, w)
    if sin_half < 1e-8:
        return np.asarray([2.0 * x, 2.0 * y, 2.0 * z], dtype=np.float32)
    axis = np.asarray([x, y, z], dtype=np.float64) / sin_half
    return (axis * angle).astype(np.float32)


def rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = rotvec / angle
    half = 0.5 * angle
    quat = np.concatenate([axis * math.sin(half), [math.cos(half)]])
    return quat.astype(np.float32)


def compose_rotvec(base_rotvec: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """Compose rotations approximately enough for small policy deltas."""
    # For small delta actions this first-order composition is usually what the policy expects.
    # If large rotation deltas are needed, replace this with scipy Rotation composition.
    return np.asarray(base_rotvec, dtype=np.float32) + np.asarray(delta_rotvec, dtype=np.float32)


def normalized_gripper_from_observation(obs: dict[str, Any], gripper_max_finger_width: float = 0.04) -> float:
    """Convert the measured per-finger width in metres to the training convention [0, 1]."""
    if gripper_max_finger_width <= 0.0:
        raise ValueError("gripper_max_finger_width must be positive")
    gripper_width = obs.get("gripper_width")
    if gripper_width is None:
        logger.warning("observation has no gripper_width; using closed (0.0)")
        return 0.0
    return float(np.clip(float(gripper_width) / gripper_max_finger_width, 0.0, 1.0))


def ros_state_from_observation(obs: dict[str, Any], gripper_max_finger_width: float = 0.04) -> np.ndarray:
    ee_pose = np.asarray(obs["ee_pose"], dtype=np.float32)
    gripper = normalized_gripper_from_observation(obs, gripper_max_finger_width)
    rotvec = quat_xyzw_to_rotvec(ee_pose[3:7])
    return np.asarray([ee_pose[0], ee_pose[1], ee_pose[2], *rotvec, gripper], dtype=np.float32)


def build_policy_input(obs: dict[str, Any], args: Args) -> dict[str, Any]:
    state = ros_state_from_observation(obs, args.gripper_max_finger_width)
    image = decode_ros_image(obs["image"])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, args.resize_size, args.resize_size))
    prompt = str(obs.get("prompt") or args.default_prompt)

    match args.input_format:
        case PolicyInputFormat.LIBERO_SINGLE_IMAGE:
            return {
                "observation/image": image,
                "observation/wrist_image": np.zeros_like(image),
                "observation/state": state,
                "prompt": prompt,
            }
        case PolicyInputFormat.RAW_MODEL:
            return {
                "image": {
                    "base_0_rgb": image,
                    "left_wrist_0_rgb": np.zeros_like(image),
                    "right_wrist_0_rgb": np.zeros_like(image),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.False_,
                    "right_wrist_0_rgb": np.False_,
                },
                "state": state,
                "prompt": prompt,
            }


def action_to_target_pose(obs: dict[str, Any], action: np.ndarray, action_format: ActionFormat) -> list[float]:
    current_pose = np.asarray(obs["ee_pose"], dtype=np.float32)
    current_state = ros_state_from_observation(obs)
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 6:
        raise ValueError(f"Policy action must have at least 6 dims, got {action.shape[0]}")

    match action_format:
        case ActionFormat.ABSOLUTE_POSE:
            if action.shape[0] < 7:
                raise ValueError("absolute_pose action_format requires 7 action dims")
            target = action[:7]
        case ActionFormat.ABSOLUTE_ROTVEC:
            quat = rotvec_to_quat_xyzw(action[3:6])
            target = np.asarray([action[0], action[1], action[2], *quat], dtype=np.float32)
        case ActionFormat.DELTA_ROTVEC:
            pos = current_pose[:3] + action[:3]
            rotvec = compose_rotvec(current_state[3:6], action[3:6])
            quat = rotvec_to_quat_xyzw(rotvec)
            target = np.asarray([pos[0], pos[1], pos[2], *quat], dtype=np.float32)

    return [float(x) for x in target]


def create_policy(args: Args) -> _policy.Policy:
    if args.policy is None:
        raise ValueError("mode=policy requires `policy:checkpoint --policy.config=... --policy.dir=...`")
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
    )


class RosWsPolicyServer:
    def __init__(self, args: Args):
        self._args = args
        self._policy = create_policy(args) if args.mode == Mode.POLICY else None

    async def run(self) -> None:
        async with _server.serve(self._handler, self._args.host, self._args.port, max_size=None):
            logger.info("ROS WebSocket server listening on ws://%s:%d", self._args.host, self._args.port)
            await asyncio.Future()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info("connection from %s opened", websocket.remote_address)
        async for raw in websocket:
            try:
                obs = json.loads(raw)
                if obs.get("type") != "observation":
                    await websocket.send(json.dumps({"type": "heartbeat"}))
                    continue
                actions = self._infer_action(obs)
                await websocket.send(
                    json.dumps(
                        {
                            "type": "action_chunk",
                            "seq": obs.get("seq"),
                            "actions": actions,
                        }
                    )
                )
            except Exception:
                logger.exception("failed to handle websocket message")
                if not self._args.hold_on_error:
                    raise
                try:
                    obs = json.loads(raw)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "action_chunk",
                                "seq": obs.get("seq"),
                                "actions": [
                                    {
                                        "target_pose": obs["ee_pose"],
                                        "gripper": normalized_gripper_from_observation(
                                            obs, self._args.gripper_max_finger_width
                                        ),
                                    }
                                ],
                            }
                        )
                    )
                except Exception:
                    await websocket.close(code=1011, reason="failed to parse observation")
                    raise
        logger.info("connection from %s closed", websocket.remote_address)

    def _infer_action(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        if self._args.mode == Mode.HOLD:
            return [
                {
                    "target_pose": [float(x) for x in obs["ee_pose"]],
                    "gripper": normalized_gripper_from_observation(obs, self._args.gripper_max_finger_width),
                }
            ]

        if self._policy is None:
            raise RuntimeError("policy was not loaded")
        policy_input = build_policy_input(obs, self._args)
        result = self._policy.infer(policy_input)
        actions = np.asarray(result["actions"])
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError(f"Policy actions must have shape [horizon, dims], got {actions.shape}")
        if actions.shape[1] < 7:
            raise ValueError(f"Franka policy actions must have 7 dims, got {actions.shape[1]}")

        action_chunk = [
            {
                "target_pose": action_to_target_pose(obs, action, self._args.action_format),
                "gripper": float(np.clip(action[6], 0.0, 1.0)),
            }
            for action in actions
        ]
        logger.info("inference seq=%s produced %d actions", obs.get("seq"), len(action_chunk))
        return action_chunk


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    server = RosWsPolicyServer(args)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("server stopped")


if __name__ == "__main__":
    main(tyro.cli(Args))
