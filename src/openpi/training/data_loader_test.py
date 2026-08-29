import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_create_torch_dataset_with_local_root(monkeypatch, tmp_path):
    calls = {}

    class FakeMetadata:
        def __init__(self, repo_id, *, root=None):
            calls["metadata"] = (repo_id, root)
            self.fps = 15

    class FakeLeRobotDataset:
        def __init__(self, repo_id, *, root=None, delta_timestamps=None):
            calls["dataset"] = (repo_id, root, delta_timestamps)

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", FakeLeRobotDataset)

    model_config = pi0_config.Pi0Config(action_horizon=3)
    data_config = _config.DataConfig(repo_id="example/local_dataset", dataset_root=str(tmp_path))
    dataset = _data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)

    expected_root = tmp_path.resolve()
    assert isinstance(dataset, FakeLeRobotDataset)
    assert calls["metadata"] == ("example/local_dataset", expected_root)
    assert calls["dataset"] == (
        "example/local_dataset",
        expected_root,
        {"actions": [0.0, 1 / 15, 2 / 15]},
    )


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
