import asyncio
import base64
import json

import numpy as np

from scripts import serve_ros_ws_policy as server


def _observation(*, gripper_width: float = 0.02) -> dict:
    return {
        "type": "observation",
        "seq": 7,
        "ee_pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        "gripper_width": gripper_width,
        "image": {
            "height": 1,
            "width": 1,
            "step": 3,
            "encoding": "rgb8",
            "data_b64": base64.b64encode(bytes([0, 0, 0])).decode("ascii"),
        },
    }


def test_gripper_width_uses_training_normalization():
    obs = _observation(gripper_width=0.02)

    assert server.normalized_gripper_from_observation(obs, 0.04) == 0.5
    np.testing.assert_allclose(server.ros_state_from_observation(obs)[-1], 0.5)


def test_build_policy_input_uses_second_camera_when_present():
    obs = _observation()
    obs["wrist_image"] = {
        "height": 1,
        "width": 1,
        "step": 3,
        "encoding": "rgb8",
        "data_b64": base64.b64encode(bytes([10, 20, 30])).decode("ascii"),
    }

    policy_input = server.build_policy_input(obs, server.Args(resize_size=4))

    assert policy_input["observation/wrist_image_mask"]
    np.testing.assert_array_equal(policy_input["observation/wrist_image"], np.full((4, 4, 3), [10, 20, 30]))


def test_gripper_normalization_clips_to_unit_interval():
    assert server.normalized_gripper_from_observation(_observation(gripper_width=-0.01), 0.04) == 0.0
    assert server.normalized_gripper_from_observation(_observation(gripper_width=0.05), 0.04) == 1.0


def test_hold_mode_returns_one_pose_and_normalized_gripper():
    policy_server = server.RosWsPolicyServer(server.Args(mode=server.Mode.HOLD))
    obs = _observation(gripper_width=0.01)

    actions = policy_server._infer_action(obs)  # noqa: SLF001

    assert actions == [{"target_pose": obs["ee_pose"], "gripper": 0.25}]


def test_policy_output_returns_complete_chunk_and_preserves_gripper():
    class FakePolicy:
        def infer(self, _inputs):
            actions = np.zeros((50, 7), dtype=np.float32)
            actions[:, 0] = np.arange(50)
            actions[:, 1:3] = [0.1, 0.2]
            actions[:, 6] = np.linspace(-0.2, 1.2, 50)
            return {"actions": actions}

    policy_server = object.__new__(server.RosWsPolicyServer)
    policy_server._args = server.Args(mode=server.Mode.POLICY)  # noqa: SLF001
    policy_server._policy = FakePolicy()  # noqa: SLF001

    actions = policy_server._infer_action(_observation())  # noqa: SLF001

    assert len(actions) == 50
    np.testing.assert_allclose(actions[0]["target_pose"], [0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(actions[-1]["target_pose"], [49.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0])
    assert actions[0]["gripper"] == 0.0
    assert actions[-1]["gripper"] == 1.0


def test_policy_output_accepts_single_action_as_one_step_chunk():
    class FakePolicy:
        def infer(self, _inputs):
            return {"actions": np.array([0.5, 0.1, 0.2, 0.0, 0.0, 0.0, 0.4])}

    policy_server = object.__new__(server.RosWsPolicyServer)
    policy_server._args = server.Args(mode=server.Mode.POLICY)  # noqa: SLF001
    policy_server._policy = FakePolicy()  # noqa: SLF001

    actions = policy_server._infer_action(_observation())  # noqa: SLF001

    assert len(actions) == 1
    assert actions[0]["gripper"] == 0.4


def test_websocket_response_uses_action_chunk_protocol():
    class FakeWebSocket:
        remote_address = ("test", 0)

        def __init__(self, incoming):
            self._incoming = iter(incoming)
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._incoming)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def send(self, message):
            self.sent.append(message)

    policy_server = server.RosWsPolicyServer(server.Args(mode=server.Mode.HOLD))
    websocket = FakeWebSocket([json.dumps(_observation())])

    asyncio.run(policy_server._handler(websocket))  # noqa: SLF001

    response = json.loads(websocket.sent[0])
    assert response["type"] == "action_chunk"
    assert response["seq"] == 7
    assert len(response["actions"]) == 1
