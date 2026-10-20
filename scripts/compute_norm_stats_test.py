import numpy as np
from scipy.spatial.transform import Rotation

from scripts.compute_norm_stats import franka_action_chunks


def test_franka_chunks_cross_pi_use_chunk_origin_and_pad_inside_episode():
    states = np.zeros((3, 7), dtype=np.float32)
    states[:, 0] = [0.0, 0.1, 0.2]
    states[:, 3:6] = Rotation.from_euler("z", [179.0, -179.0, -176.0], degrees=True).as_rotvec()
    states[:, 6] = [0.2, 0.4, 0.6]
    actions = np.concatenate([states[1:], states[-1:]])
    original = actions.copy()
    chunks = franka_action_chunks(states, actions, 0, 3, 4)
    assert chunks.shape == (3, 4, 7)
    np.testing.assert_allclose(chunks[0, :, 0], [0.1, 0.2, 0.2, 0.2], atol=1e-6)
    np.testing.assert_allclose(np.rad2deg(chunks[0, :, 5]), [2.0, 5.0, 5.0, 5.0], atol=1e-4)
    np.testing.assert_allclose(chunks[0, :, 6], [0.4, 0.6, 0.6, 0.6], atol=1e-6)
    np.testing.assert_allclose(chunks[-1, :, :6], 0.0, atol=1e-6)
    np.testing.assert_allclose(chunks[-1, :, 6], 0.6, atol=1e-6)
    np.testing.assert_array_equal(actions, original)
