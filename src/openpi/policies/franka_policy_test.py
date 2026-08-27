import numpy as np
from scipy.spatial.transform import Rotation

from openpi.policies import franka_policy


def test_relative_rotation_uses_short_path_across_rotvec_boundary():
    state = np.array([0.5, -0.1, 0.2, 0.0, 0.0, np.deg2rad(179.0), 0.25])
    actions = np.array([[0.6, -0.3, 0.5, 0.0, 0.0, np.deg2rad(-179.0), 0.75]])

    transformed = franka_policy.FrankaRelativeActions()({"state": state, "actions": actions})

    np.testing.assert_allclose(transformed["actions"][0, :3], [0.1, -0.2, 0.3], atol=1e-7)
    np.testing.assert_allclose(transformed["actions"][0, 3:6], [0.0, 0.0, np.deg2rad(2.0)], atol=1e-7)
    np.testing.assert_allclose(transformed["actions"][0, 6], 0.75)


def test_relative_and_absolute_pose_transforms_round_trip():
    state = np.array([0.5, -0.1, 0.2, 0.2, -0.3, 3.0, 0.25])
    actions = np.array(
        [
            [0.6, -0.3, 0.5, -0.2, 0.3, -3.0, 0.75],
            [0.4, 0.2, 0.1, 0.8, -0.4, 0.1, 0.10],
        ]
    )

    relative = franka_policy.FrankaRelativeActions()({"state": state, "actions": actions.copy()})
    recovered = franka_policy.FrankaAbsoluteActions()(relative)["actions"]

    np.testing.assert_allclose(recovered[:, :3], actions[:, :3], atol=1e-7)
    np.testing.assert_allclose(recovered[:, 6:], actions[:, 6:], atol=1e-7)
    expected_rotations = Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
    recovered_rotations = Rotation.from_rotvec(recovered[:, 3:6]).as_matrix()
    np.testing.assert_allclose(recovered_rotations, expected_rotations, atol=1e-7)


def test_franka_pose_transforms_are_noop_without_actions():
    item = {"state": np.zeros(7)}

    assert franka_policy.FrankaRelativeActions()(item) is item
    assert franka_policy.FrankaAbsoluteActions()(item) is item
