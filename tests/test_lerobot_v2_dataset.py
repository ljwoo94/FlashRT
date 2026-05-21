import numpy as np
import sys
import types


def test_load_calibration_obs_stratifies_by_episode_and_normalizes_rows():
    from flash_rt.datasets.lerobot_v2 import (
        DEFAULT_RBY1_IMAGE_KEYS,
        load_calibration_obs,
    )

    rows = []
    for ep in range(3):
        for frame in range(4):
            value = ep * 10 + frame
            row = {
                "episode_index": ep,
                "frame_index": frame,
                "observation.state": np.full((44,), value, dtype=np.float32),
            }
            for key in DEFAULT_RBY1_IMAGE_KEYS:
                row[key] = np.full((3, 8, 10), value, dtype=np.uint8)
            rows.append(row)

    obs = load_calibration_obs(rows, n=6)

    assert [int(o["state"][0]) for o in obs] == [0, 10, 20, 3, 13, 23]
    assert len(obs) == 6
    for item in obs:
        assert set(item) == {"images", "state"}
        assert len(item["images"]) == 4
        assert item["state"].shape == (44,)
        assert item["state"].dtype == np.float32
        for image in item["images"]:
            assert image.shape == (224, 224, 3)
            assert image.dtype == np.uint8


def test_load_calibration_obs_uses_even_global_sampling_without_episode_key():
    from flash_rt.datasets.lerobot_v2 import (
        DEFAULT_RBY1_IMAGE_KEYS,
        load_calibration_obs,
    )

    rows = []
    for frame in range(5):
        row = {
            "frame_index": frame,
            "observation.state": np.array([frame], dtype=np.float32),
        }
        for key in DEFAULT_RBY1_IMAGE_KEYS:
            row[key] = np.full((6, 7, 3), frame, dtype=np.uint8)
        rows.append(row)

    obs = load_calibration_obs(rows, n=3)

    assert [int(o["state"][0]) for o in obs] == [0, 2, 4]


def test_load_calibration_obs_spreads_many_episodes():
    from flash_rt.datasets.lerobot_v2 import (
        DEFAULT_RBY1_IMAGE_KEYS,
        load_calibration_obs,
    )

    rows = []
    for ep in range(10):
        for frame in range(2):
            row = {
                "episode_index": ep,
                "frame_index": frame,
                "observation.state": np.array([ep], dtype=np.float32),
            }
            for key in DEFAULT_RBY1_IMAGE_KEYS:
                row[key] = np.full((3, 5, 4), ep, dtype=np.uint8)
            rows.append(row)

    obs = load_calibration_obs(rows, n=4)

    picked_episodes = [int(o["state"][0]) for o in obs]
    assert len(picked_episodes) == 4
    assert picked_episodes[:3] == [0, 3, 6]


def test_load_calibration_obs_accepts_single_image_key_mapping():
    from flash_rt.datasets.lerobot_v2 import load_calibration_obs

    rows = []
    for frame in range(2):
        rows.append(
            {
                "episode_index": 0,
                "frame_index": frame,
                "observation.state": [float(frame)],
                "cam": np.full((3, 5, 4), frame, dtype=np.uint8),
            }
        )

    obs = load_calibration_obs(rows, n=1, image_keys=("cam",))

    assert set(obs[0]) == {"image", "state"}
    assert obs[0]["image"].shape == (224, 224, 3)
    assert obs[0]["image"].dtype == np.uint8


def test_import_lerobot_dataset_prefers_current_path():
    from flash_rt.datasets.lerobot_v2 import _import_lerobot_dataset

    sentinel = object()
    module = types.ModuleType("lerobot.datasets")
    module.LeRobotDataset = sentinel

    old_lerobot = sys.modules.get("lerobot")
    old_datasets = sys.modules.get("lerobot.datasets")
    pkg = types.ModuleType("lerobot")
    pkg.__path__ = []
    sys.modules["lerobot"] = pkg
    sys.modules["lerobot.datasets"] = module
    try:
        assert _import_lerobot_dataset() is sentinel
    finally:
        if old_lerobot is None:
            sys.modules.pop("lerobot", None)
        else:
            sys.modules["lerobot"] = old_lerobot
        if old_datasets is None:
            sys.modules.pop("lerobot.datasets", None)
        else:
            sys.modules["lerobot.datasets"] = old_datasets
