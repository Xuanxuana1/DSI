from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class NPZExposureDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        max_rows: int | None = None,
        content_arrays: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        split_path = Path(path)
        mmap_dir = split_path.with_suffix("")
        if mmap_dir.is_dir():

            array_paths = sorted(mmap_dir.glob("*.npy"))
            self.arrays = {
                item.stem: np.load(item, mmap_mode="r", allow_pickle=False)
                for item in array_paths
            }
            if "event" not in self.arrays:
                raise KeyError(f"Missing event.npy in mmap split: {mmap_dir}")
            length = len(self.arrays["event"])
        else:

            loaded = np.load(split_path, allow_pickle=False)
            length = len(loaded["event"])
            if max_rows is not None:
                length = min(length, int(max_rows))
            self.arrays = {key: loaded[key][:length] for key in loaded.files}
        if max_rows is not None:
            length = min(length, int(max_rows))
            self.arrays = {key: value[:length] for key, value in self.arrays.items()}
        if "watch_event" not in self.arrays:
            self.arrays["watch_event"] = np.minimum(self.arrays["event"], 2).astype(np.int64)
        if "raw_sparse" not in self.arrays:
            self.arrays["raw_sparse"] = np.zeros((length, 0), dtype=np.int64)
        if "raw_numeric" not in self.arrays:
            self.arrays["raw_numeric"] = np.zeros((length, 0), dtype=np.float32)
        if "raw_seq" not in self.arrays:
            self.arrays["raw_seq"] = np.zeros((length, 0), dtype=np.int64)
        if "raw_seq_mask" not in self.arrays:
            self.arrays["raw_seq_mask"] = np.zeros((length, 0), dtype=np.float32)
        content_path = split_path.parent / "item_content.npz"
        if content_arrays is not None:

            self.content_category = content_arrays[0]
            self.content_numeric = content_arrays[1]
        elif content_path.exists():
            content = np.load(content_path, allow_pickle=False)
            self.content_category = content["content_category"]
            self.content_numeric = content["content_numeric"]
        else:
            self.content_category = np.zeros((0, 0), dtype=np.int64)
            self.content_numeric = np.zeros((0, 0), dtype=np.float32)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def _lookup_from_indices(self, indices: Sequence[int] | np.ndarray | slice | range) -> slice | np.ndarray:

        if isinstance(indices, slice):
            return indices

        if isinstance(indices, range):
            return slice(indices.start, indices.stop, indices.step)

        if isinstance(indices, torch.Tensor):
            values = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            values = np.asarray(indices, dtype=np.int64)
        if values.ndim == 0:
            return np.asarray([int(values)], dtype=np.int64)
        if len(values) == 0:
            return values
        if len(values) == 1:
            return slice(int(values[0]), int(values[0]) + 1)
        if np.all(values[1:] == values[:-1] + 1):
            return slice(int(values[0]), int(values[-1]) + 1)
        return values

    def _array_to_tensor(self, key: str, lookup: slice | np.ndarray, dtype: torch.dtype) -> torch.Tensor:

        value = np.array(self.arrays[key][lookup], copy=True)

        return torch.as_tensor(value, dtype=dtype)

    def __getitems__(self, indices: Sequence[int] | np.ndarray | slice | range) -> dict[str, torch.Tensor]:

        lookup = self._lookup_from_indices(indices)

        item_idx = np.array(self.arrays["item_idx"][lookup], copy=True).astype(np.int64, copy=False)
        if len(self.content_category) > 0:
            content_category = np.array(self.content_category[item_idx], copy=True)
            content_numeric = np.array(self.content_numeric[item_idx], copy=True)
        else:
            batch_size = len(item_idx)
            content_category = np.zeros((batch_size, 0), dtype=np.int64)
            content_numeric = np.zeros((batch_size, 0), dtype=np.float32)

        return {
            "user_idx": self._array_to_tensor("user_idx", lookup, torch.long),
            "item_idx": torch.as_tensor(item_idx, dtype=torch.long),
            "tab": self._array_to_tensor("tab", lookup, torch.long),
            "prev_event": self._array_to_tensor("prev_event", lookup, torch.long),
            "numeric": self._array_to_tensor("numeric", lookup, torch.float32),
            "session_numeric": self._array_to_tensor("session_numeric", lookup, torch.float32),
            "content_category": torch.as_tensor(content_category, dtype=torch.long),
            "content_numeric": torch.as_tensor(content_numeric, dtype=torch.float32),
            "raw_sparse": self._array_to_tensor("raw_sparse", lookup, torch.long),
            "raw_numeric": self._array_to_tensor("raw_numeric", lookup, torch.float32),
            "raw_seq": self._array_to_tensor("raw_seq", lookup, torch.long),
            "raw_seq_mask": self._array_to_tensor("raw_seq_mask", lookup, torch.float32),
            "duration_sec": self._array_to_tensor("duration_sec", lookup, torch.float32),
            "event": self._array_to_tensor("event", lookup, torch.long),
            "watch_event": self._array_to_tensor("watch_event", lookup, torch.long),
            "time_bin": self._array_to_tensor("time_bin", lookup, torch.long),
            "engagement_mark": self._array_to_tensor("engagement_mark", lookup, torch.float32),
            "play_time_sec": self._array_to_tensor("play_time_sec", lookup, torch.float32),
            "session_key": self._array_to_tensor("session_key", lookup, torch.long),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item_idx = int(self.arrays["item_idx"][idx])
        if len(self.content_category) > 0:
            content_category = self.content_category[item_idx]
            content_numeric = self.content_numeric[item_idx]
        else:
            content_category = np.zeros((0,), dtype=np.int64)
            content_numeric = np.zeros((0,), dtype=np.float32)
        return {
            "user_idx": torch.tensor(self.arrays["user_idx"][idx], dtype=torch.long),
            "item_idx": torch.tensor(item_idx, dtype=torch.long),
            "tab": torch.tensor(self.arrays["tab"][idx], dtype=torch.long),
            "prev_event": torch.tensor(self.arrays["prev_event"][idx], dtype=torch.long),
            "numeric": torch.tensor(self.arrays["numeric"][idx], dtype=torch.float32),
            "session_numeric": torch.tensor(self.arrays["session_numeric"][idx], dtype=torch.float32),
            "content_category": torch.tensor(content_category, dtype=torch.long),
            "content_numeric": torch.tensor(content_numeric, dtype=torch.float32),
            "raw_sparse": torch.tensor(self.arrays["raw_sparse"][idx], dtype=torch.long),
            "raw_numeric": torch.tensor(self.arrays["raw_numeric"][idx], dtype=torch.float32),
            "raw_seq": torch.tensor(self.arrays["raw_seq"][idx], dtype=torch.long),
            "raw_seq_mask": torch.tensor(self.arrays["raw_seq_mask"][idx], dtype=torch.float32),
            "duration_sec": torch.tensor(self.arrays["duration_sec"][idx], dtype=torch.float32),
            "event": torch.tensor(self.arrays["event"][idx], dtype=torch.long),
            "watch_event": torch.tensor(self.arrays["watch_event"][idx], dtype=torch.long),
            "time_bin": torch.tensor(self.arrays["time_bin"][idx], dtype=torch.long),
            "engagement_mark": torch.tensor(self.arrays["engagement_mark"][idx], dtype=torch.float32),
            "play_time_sec": torch.tensor(self.arrays["play_time_sec"][idx], dtype=torch.float32),
            "session_key": torch.tensor(self.arrays["session_key"][idx], dtype=torch.long),
        }


class BlockShuffleBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        length: int,
        batch_size: int,
        block_rows: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:

        self.length = int(length)
        self.batch_size = int(batch_size)
        self.block_rows = max(int(block_rows), int(batch_size))
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:

        rng = np.random.default_rng(self.seed + self.epoch)
        block_starts = np.arange(0, self.length, self.block_rows, dtype=np.int64)
        rng.shuffle(block_starts)
        self.epoch += 1

        global_batch_idx = 0

        for block_start in block_starts:
            block_end = min(int(block_start) + self.block_rows, self.length)
            for batch_start in range(int(block_start), block_end, self.batch_size):
                batch_end = min(batch_start + self.batch_size, block_end)
                if global_batch_idx % self.world_size == self.rank:
                    yield list(range(batch_start, batch_end))
                global_batch_idx += 1

    def __len__(self) -> int:
        total_batches = int(math.ceil(self.length / self.batch_size))
        return len(range(self.rank, total_batches, self.world_size))


def identity_collate(batch: Any) -> Any:
    return batch


def load_manifest(processed_dir: str | Path) -> dict[str, Any]:
    with (Path(processed_dir) / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
