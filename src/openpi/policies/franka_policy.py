import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka sponge policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the sponge",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Inputs for a one- or two-camera Franka absolute-pose policy.

    Expected pre-repack keys:
      observation/image: RGB image
      observation/wrist_image: optional second RGB camera
      observation/state: [x,y,z,rx,ry,rz,gripper]
      actions: [x,y,z,rx,ry,rz,gripper]
      prompt: language task
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        has_wrist_image = "observation/wrist_image" in data
        wrist_image = _parse_image(data["observation/wrist_image"]) if has_wrist_image else np.zeros_like(base_image)
        if "observation/wrist_image_mask" in data:
            has_wrist_image = bool(np.asarray(data["observation/wrist_image_mask"]).item())

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.bool_(has_wrist_image), np.False_)
            case _model.ModelType.PI0_FAST:
                image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.bool_(has_wrist_image), np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}


@dataclasses.dataclass(frozen=True)
class FrankaRelativeActions(transforms.DataTransformFn):
    """Convert absolute Franka poses to pose deltas relative to the current state.

    Translation is expressed in the world frame. Rotation is composed on SO(3), rather
    than subtracting rotation-vector components, so crossing the +/-pi rotvec boundary
    produces the short relative rotation. The gripper and any padded dimensions remain
    absolute.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        if state.ndim != 1 or state.shape[-1] < 6 or actions.shape[-1] < 6:
            raise ValueError("Franka pose transforms require a 1D state and actions with at least 6 dimensions.")

        actions[..., :3] -= state[:3]
        current_rotation = Rotation.from_rotvec(state[3:6])
        target_rotations = Rotation.from_rotvec(actions[..., 3:6])
        actions[..., 3:6] = (current_rotation.inv() * target_rotations).as_rotvec()
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class FrankaAbsoluteActions(transforms.DataTransformFn):
    """Convert Franka pose deltas back to absolute targets for robot execution."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data

        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"]).copy()
        if state.ndim != 1 or state.shape[-1] < 6 or actions.shape[-1] < 6:
            raise ValueError("Franka pose transforms require a 1D state and actions with at least 6 dimensions.")

        actions[..., :3] += state[:3]
        current_rotation = Rotation.from_rotvec(state[3:6])
        relative_rotations = Rotation.from_rotvec(actions[..., 3:6])
        actions[..., 3:6] = (current_rotation * relative_rotations).as_rotvec()
        data["actions"] = actions
        return data
