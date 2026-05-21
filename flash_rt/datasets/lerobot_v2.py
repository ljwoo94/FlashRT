"""Generic LeRobot v2 calibration loader for FlashRT RB-Y1.

The main entry point is :func:`load_calibration_obs`. It returns plain
Python observation dicts ready for the Pi0.5 RB-Y1 Thor frontend::

    obs_list = load_calibration_obs("/path/or/repo_id", n=32)
    pipe.calibrate(obs_list, percentile=99.9)

For the default RB-Y1 four-view schema, each observation is shaped as::

    {"images": [high_left, high_right, left_wrist, right_wrist],
     "state":  float32[state_dim]}

Images are HWC uint8 RGB arrays resized to ``image_size x image_size``.
The loader is torch/CUDA-free. Local LeRobot-v2 roots are read directly
from ``meta/info.json`` + parquet files. Repo ids use the LeRobot 0.1.x
``lerobot.common.datasets.lerobot_dataset.LeRobotDataset`` API.
"""

from __future__ import annotations

import io
import json
import logging
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_RBY1_IMAGE_KEYS: Tuple[str, str, str, str] = (
    "observation.images.cam_high_left",
    "observation.images.cam_high_right",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def load_calibration_obs(
    dataset: Any,
    *,
    n: int = 32,
    image_keys: Sequence[str] = DEFAULT_RBY1_IMAGE_KEYS,
    state_key: str = "observation.state",
    image_size: int = 224,
    episode_key: str = "episode_index",
    frame_key: str = "frame_index",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Sample and load LeRobot v2 observations for frontend calibration.

    Args:
        dataset: A LeRobotDataset-like object, a sequence of row dicts, or a
            local path/repo id accepted by ``LeRobotDataset``.
        n: Number of observations to return. If ``n`` exceeds dataset length,
            all available rows are returned once.
        image_keys: Image columns to decode. With the default four RB-Y1 keys,
            output observations use ``{"images": [..], "state": state}`` in
            the expected high-left, high-right, left-wrist, right-wrist order.
            With a single image key, output uses ``{"image": image, ...}``.
        state_key: State column to convert to float32.
        image_size: Target square image size. Output images are HWC uint8 RGB
            arrays with shape ``(image_size, image_size, 3)``.
        episode_key: Episode/video column for deterministic stratified
            sampling. If absent, rows are sampled evenly across the full
            dataset.
        frame_key: Optional frame column used to keep samples ordered within
            each episode when present.
        verbose: Log selected row indices.

    Returns:
        A list of calibration observations suitable for
        ``pipe.calibrate(obs_list, percentile=...)``.
    """
    if n <= 0:
        return []
    if not image_keys:
        raise ValueError("image_keys must contain at least one key")

    ds = _resolve_dataset(dataset)
    indices = _sample_indices(ds, n=n, episode_key=episode_key, frame_key=frame_key)
    if verbose:
        logger.info(
            "LeRobot v2 calibration sample: %d/%d rows from %s",
            len(indices),
            n,
            _describe_dataset(ds),
        )
        logger.info("picked row indices: %s", indices)

    return [
        _row_to_obs(
            _get_row(ds, i),
            image_keys=image_keys,
            state_key=state_key,
            image_size=image_size,
        )
        for i in indices
    ]


def _resolve_dataset(dataset: Any) -> Any:
    if isinstance(dataset, (str, pathlib.Path)):
        path = pathlib.Path(dataset)
        if (path / "meta" / "info.json").exists():
            return _LocalLeRobotV2Dataset(path)
        return _open_lerobot_dataset(dataset)
    return dataset


def _open_lerobot_dataset(dataset: Union[str, pathlib.Path]) -> Any:
    LeRobotDataset = _import_lerobot_dataset()
    if LeRobotDataset is None:
        raise ImportError(
            "Loading a LeRobot v2 path or repo id requires the optional "
            "'lerobot' package. Install LeRobot or pass an already-open "
            "dataset/sequence of rows to load_calibration_obs()."
        )

    value = str(dataset)
    try:
        return LeRobotDataset(value)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open LeRobotDataset for repo_id/path {value!r}. "
            "This loader targets LeRobot v2 datasets with lerobot==0.1.0; "
            "for a local dataset root, pass the directory containing "
            "meta/info.json."
        ) from exc


def _import_lerobot_dataset() -> Any:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        pass
    try:
        from lerobot.datasets import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        return None


class _LocalLeRobotV2Dataset:
    """Minimal local reader for LeRobot-v2 roots."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        info_path = root / "meta" / "info.json"
        with open(info_path, encoding="utf-8") as f:
            self.info = json.load(f)
        self._rows = self._read_rows()
        self.column_names = list(self._rows[0].keys()) if self._rows else []

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return [row[key] for row in self._rows]
        return self._rows[int(key)]

    def _read_rows(self) -> List[Dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading a local LeRobot v2 root requires pyarrow. "
                "Install pyarrow or pass an already-open LeRobotDataset."
            ) from exc

        data_path = self.info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
        chunks_size = int(self.info.get("chunks_size", 1000))
        total_episodes = int(self.info.get("total_episodes", 0))
        rows: List[Dict[str, Any]] = []

        if total_episodes > 0:
            paths = [
                self.root / data_path.format(
                    episode_chunk=ep // chunks_size,
                    episode_index=ep,
                )
                for ep in range(total_episodes)
            ]
        else:
            paths = sorted((self.root / "data").glob("**/*.parquet"))

        for path in paths:
            if not path.exists():
                continue
            table = pq.read_table(path)
            rows.extend(table.to_pylist())

        if not rows:
            raise RuntimeError(f"no parquet rows found under {self.root!s}")
        return rows


def _sample_indices(
    dataset: Any,
    *,
    n: int,
    episode_key: str,
    frame_key: str,
) -> List[int]:
    length = len(dataset)
    if length == 0:
        return []
    count = min(int(n), int(length))

    episode_values = _column_values(dataset, episode_key)
    if episode_values is None:
        return _evenly_spaced_indices(length, count)

    groups: Dict[Any, List[int]] = defaultdict(list)
    for idx, ep in enumerate(episode_values):
        groups[_scalar(ep)].append(idx)
    if not groups:
        return _evenly_spaced_indices(length, count)

    frame_values = _column_values(dataset, frame_key)
    if frame_values is not None:
        for indices in groups.values():
            indices.sort(key=lambda i: (_scalar(frame_values[i]), i))

    episodes = sorted(groups, key=lambda ep: str(ep))
    n_eps = min(len(episodes), max(count // 2, 3))
    ep_stride = max(1, len(episodes) // n_eps)
    chosen_episodes = episodes[::ep_stride][:n_eps]
    frames_per_ep = max(1, -(-count // max(len(chosen_episodes), 1)))

    per_episode_picks = [
        _pick_from_group(groups[ep], frames_per_ep)
        for ep in chosen_episodes
    ]

    selected: List[int] = []
    for offset in range(frames_per_ep):
        for picks in per_episode_picks:
            if offset >= len(picks):
                continue
            selected.append(picks[offset])
            if len(selected) >= count:
                break
        if len(selected) >= count:
            break

    if len(selected) < count:
        seen = set(selected)
        for ep in episodes:
            for idx in groups[ep]:
                if idx in seen:
                    continue
                selected.append(idx)
                seen.add(idx)
                if len(selected) >= count:
                    break
            if len(selected) >= count:
                break

    return selected[:count]


def _column_values(dataset: Any, key: str) -> Union[List[Any], None]:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    for source in (dataset, hf_dataset):
        if source is None:
            continue
        column_names = getattr(source, "column_names", None)
        if column_names is not None and key not in column_names:
            continue
        try:
            values = source[key]
        except (KeyError, TypeError, AttributeError):
            continue
        return list(values)

    values = []
    for idx in range(len(dataset)):
        row = _get_row(dataset, idx)
        if not _has_key(row, key):
            return None
        values.append(_get_value(row, key))
    return values


def _evenly_spaced_indices(length: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return [int(round(x)) for x in np.linspace(0, length - 1, count)]


def _pick_from_group(indices: Sequence[int], count: int) -> List[int]:
    if count >= len(indices):
        return list(indices)
    positions = _evenly_spaced_indices(len(indices), count)
    return [indices[pos] for pos in positions]


def _get_row(dataset: Any, index: int) -> Any:
    if hasattr(dataset, "__getitem__"):
        return dataset[index]
    raise TypeError(
        f"dataset object of type {type(dataset).__name__!r} does not support indexing"
    )


def _row_to_obs(
    row: Any,
    *,
    image_keys: Sequence[str],
    state_key: str,
    image_size: int,
) -> Dict[str, Any]:
    images = [
        _coerce_image(_get_value(row, key), target_size=image_size)
        for key in image_keys
    ]
    obs: Dict[str, Any]
    if len(images) == 1:
        obs = {"image": images[0]}
    else:
        obs = {"images": images}
    obs["state"] = np.asarray(_get_value(row, state_key), dtype=np.float32)
    return obs


def _coerce_image(cell: Any, *, target_size: int) -> np.ndarray:
    """Convert a LeRobot image cell to HWC uint8 RGB at ``target_size``."""
    if isinstance(cell, Mapping):
        if "bytes" in cell and cell["bytes"] is not None:
            return _decode_image_bytes(cell["bytes"], target_size=target_size)
        if "path" in cell and cell["path"]:
            return _decode_image_path(cell["path"], target_size=target_size)

    if isinstance(cell, (bytes, bytearray)):
        return _decode_image_bytes(cell, target_size=target_size)

    if _looks_like_pil_image(cell):
        return _resize_array(np.asarray(cell.convert("RGB")), target_size=target_size)

    arr = _as_numpy(cell)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim != 3:
        raise ValueError(
            f"image array must have rank 2 or 3, got shape {arr.shape!r}"
        )

    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(
            f"image array must have 1, 3, or 4 channels, got shape {arr.shape!r}"
        )

    arr = _to_uint8(arr)
    return _resize_array(arr, target_size=target_size)


def _decode_image_bytes(raw: Union[bytes, bytearray], *, target_size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    return _pil_to_array(img, target_size=target_size)


def _decode_image_path(path: Union[str, pathlib.Path], *, target_size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return _pil_to_array(img, target_size=target_size)


def _pil_to_array(img: Any, *, target_size: int) -> np.ndarray:
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), _pil_bilinear())
    return np.asarray(img, dtype=np.uint8)


def _resize_array(arr: np.ndarray, *, target_size: int) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.shape[:2] == (target_size, target_size):
        return arr
    try:
        from PIL import Image
    except ImportError:
        return _resize_nearest(arr, target_size=target_size)
    img = Image.fromarray(arr, mode="RGB")
    return _pil_to_array(img, target_size=target_size)


def _resize_nearest(arr: np.ndarray, *, target_size: int) -> np.ndarray:
    src_h, src_w = arr.shape[:2]
    y = np.linspace(0, src_h - 1, target_size).round().astype(np.int64)
    x = np.linspace(0, src_w - 1, target_size).round().astype(np.int64)
    return np.ascontiguousarray(arr[y][:, x], dtype=np.uint8)


def _pil_bilinear() -> int:
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        finite = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        if finite.size and finite.max() <= 1.0 and finite.min() >= 0.0:
            finite = finite * 255.0
        return np.clip(finite, 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except TypeError:
            pass
    return np.asarray(value)


def _looks_like_pil_image(value: Any) -> bool:
    return hasattr(value, "convert") and hasattr(value, "size")


def _has_key(row: Any, key: str) -> bool:
    if isinstance(row, Mapping):
        return key in row
    if hasattr(row, "index"):
        try:
            return key in row.index
        except TypeError:
            return False
    return hasattr(row, key)


def _get_value(row: Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row[key]
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        if hasattr(row, key):
            return getattr(row, key)
        raise KeyError(f"row is missing required key {key!r}")


def _scalar(value: Any) -> Any:
    arr = _as_numpy(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(()).item()
    return value


def _describe_dataset(dataset: Any) -> str:
    root = getattr(dataset, "root", None)
    repo_id = getattr(dataset, "repo_id", None)
    if repo_id is not None:
        return str(repo_id)
    if root is not None:
        return str(root)
    return type(dataset).__name__
