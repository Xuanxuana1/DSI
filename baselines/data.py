from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import duration_quantile_edges

logger = logging.getLogger(__name__)


class DSIBaselineDataset(Dataset):
    def __init__(self, split_path: str | Path, max_rows: int | None = None, include_raw_features: bool = False) -> None:
        logger.info("Loading DSI baseline data. split=%s max_rows=%s include_raw=%s", split_path, max_rows, include_raw_features)

        split = Path(split_path)
        arrays = np.load(split, allow_pickle=False)
        length = len(arrays["play_time_sec"])
        if max_rows is not None:
            length = min(length, int(max_rows))

        self.include_raw_features = bool(include_raw_features)
        self.user_id = np.asarray(arrays["user_idx"][:length], dtype=np.int64)
        self.video_id = np.asarray(arrays["item_idx"][:length], dtype=np.int64)
        self.tab = np.asarray(arrays["tab"][:length], dtype=np.int64)
        self.prev_event = np.asarray(arrays["prev_event"][:length], dtype=np.int64)
        self.numeric = np.asarray(arrays["numeric"][:length], dtype=np.float32)
        self.session_numeric = np.asarray(arrays["session_numeric"][:length], dtype=np.float32)
        self.duration = np.asarray(arrays["duration_sec"][:length], dtype=np.float32)
        self.play_time = np.asarray(arrays["play_time_sec"][:length], dtype=np.float32)
        self.duration_bucket = np.zeros(length, dtype=np.int64)
        self.event = np.asarray(arrays["event"][:length], dtype=np.int64)
        self.engagement_mark = np.asarray(arrays["engagement_mark"][:length], dtype=np.float32) if "engagement_mark" in arrays else np.zeros(length, dtype=np.float32)
        self.session_key = np.asarray(arrays["session_key"][:length], dtype=np.int64)

        if self.include_raw_features:
            self.raw_sparse = np.asarray(arrays["raw_sparse"][:length], dtype=np.int64) if "raw_sparse" in arrays else np.zeros((length, 0), dtype=np.int64)
            self.raw_numeric = np.asarray(arrays["raw_numeric"][:length], dtype=np.float32) if "raw_numeric" in arrays else np.zeros((length, 0), dtype=np.float32)
            self.raw_seq = np.asarray(arrays["raw_seq"][:length], dtype=np.int64) if "raw_seq" in arrays else np.zeros((length, 0), dtype=np.int64)
            self.raw_seq_mask = np.asarray(arrays["raw_seq_mask"][:length], dtype=np.float32) if "raw_seq_mask" in arrays else np.zeros((length, 0), dtype=np.float32)
        else:
            self.raw_sparse = np.zeros((length, 0), dtype=np.int64)
            self.raw_numeric = np.zeros((length, 0), dtype=np.float32)
            self.raw_seq = np.zeros((length, 0), dtype=np.int64)
            self.raw_seq_mask = np.zeros((length, 0), dtype=np.float32)

        content_path = split.parent / "item_content.npz"
        if content_path.exists():
            content = np.load(content_path, allow_pickle=False)
            content_category_all = content["content_category"]
            content_numeric_all = content["content_numeric"]
            self.content_category = np.asarray(content_category_all[self.video_id], dtype=np.int64)
            self.content_numeric = np.asarray(content_numeric_all[self.video_id], dtype=np.float32)
        else:
            self.content_category = np.zeros((length, 0), dtype=np.int64)
            self.content_numeric = np.zeros((length, 0), dtype=np.float32)

        self._ctn_seconds = np.concatenate(
            [
                self.numeric,
                self.session_numeric,
                np.log1p(self.duration).reshape(-1, 1),
                self.raw_numeric,
                self.content_numeric,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        self.duration_model = np.zeros_like(self.duration, dtype=np.float32)
        self.ctn_model = np.zeros_like(self._ctn_seconds, dtype=np.float32)
        logger.info("DSI baseline data loaded. split=%s rows=%d ctn_dim=%d raw_sparse_dim=%d", split_path, len(self), self.ctn_model.shape[1], self.raw_sparse.shape[1])

    def set_model_continuous_scale(self, label_scale: float) -> None:
        logger.info("Building model-side normalized continuous features. label_scale=%.6f", float(label_scale))

        scale = max(float(label_scale), 1e-6)
        self.duration_model = (self.duration / scale).astype(np.float32, copy=False)

        model_ctn = np.array(self._ctn_seconds, dtype=np.float32, copy=True)
        numeric_dim = int(self.numeric.shape[1])
        session_dim = int(self.session_numeric.shape[1])

        if numeric_dim >= 5:
            model_ctn[:, 0] = np.log1p(self.duration_model).astype(np.float32, copy=False)
            for col_idx in (2, 3, 4):
                raw_seconds = np.expm1(model_ctn[:, col_idx].astype(np.float64, copy=False))
                model_ctn[:, col_idx] = np.log1p(np.clip(raw_seconds / scale, 0.0, None)).astype(np.float32, copy=False)

        session_offset = numeric_dim
        if session_dim >= 4:
            for local_idx in (1, 2, 3):
                col_idx = session_offset + local_idx
                raw_seconds = np.expm1(model_ctn[:, col_idx].astype(np.float64, copy=False))
                model_ctn[:, col_idx] = np.log1p(np.clip(raw_seconds / scale, 0.0, None)).astype(np.float32, copy=False)

        appended_log_duration_idx = numeric_dim + session_dim
        if appended_log_duration_idx < model_ctn.shape[1]:
            model_ctn[:, appended_log_duration_idx] = np.log1p(self.duration_model).astype(np.float32, copy=False)
        self.ctn_model = model_ctn
        logger.info("Model-side normalized continuous features built. ctn_dim=%d", int(self.ctn_model.shape[1]))

    def __len__(self) -> int:
        return int(self.play_time.shape[0])

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

        features = {
            "user_id": torch.tensor(self.user_id[idx], dtype=torch.long),
            "video_id": torch.tensor(self.video_id[idx], dtype=torch.long),
            "tab": torch.tensor(self.tab[idx], dtype=torch.long),
            "prev_event": torch.tensor(self.prev_event[idx], dtype=torch.long),
            "duration": torch.tensor(self.duration[idx], dtype=torch.float32),
            "duration_model": torch.tensor(self.duration_model[idx], dtype=torch.float32),
            "duration_bucket": torch.tensor(self.duration_bucket[idx], dtype=torch.long),
            "ctn_model": torch.tensor(self.ctn_model[idx], dtype=torch.float32),
            "raw_sparse": torch.tensor(self.raw_sparse[idx], dtype=torch.long),
            "raw_seq": torch.tensor(self.raw_seq[idx], dtype=torch.long),
            "raw_seq_mask": torch.tensor(self.raw_seq_mask[idx], dtype=torch.float32),
            "content_category": torch.tensor(self.content_category[idx], dtype=torch.long),
            "event": torch.tensor(self.event[idx], dtype=torch.long),
            "session_key": torch.tensor(self.session_key[idx], dtype=torch.long),
        }

        label = torch.tensor(self.play_time[idx], dtype=torch.float32)
        return features, label


class DSIBaselineBatchLoader:
    def __init__(
        self,
        dataset: DSIBaselineDataset,
        device: torch.device,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        logger.info("Initializing the batch data iterator. rows=%d batch_size=%d shuffle=%s", len(dataset), int(batch_size), shuffle)

        self.dataset = dataset
        self.device = device
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        logger.info("Batch data iterator initialized. rows=%d", len(dataset))

    def __len__(self) -> int:

        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        logger.info("Iterating over baseline batches. rows=%d shuffle=%s", len(self.dataset), self.shuffle)

        row_count = len(self.dataset)
        if self.shuffle:
            indices = np.random.permutation(row_count)
        else:
            indices = None

        for start in range(0, row_count, self.batch_size):
            end = min(start + self.batch_size, row_count)
            if indices is None:
                batch_index = slice(start, end)
            else:
                batch_index = indices[start:end]

            features = {
                "user_id": torch.as_tensor(self.dataset.user_id[batch_index], dtype=torch.long, device=self.device),
                "video_id": torch.as_tensor(self.dataset.video_id[batch_index], dtype=torch.long, device=self.device),
                "tab": torch.as_tensor(self.dataset.tab[batch_index], dtype=torch.long, device=self.device),
                "prev_event": torch.as_tensor(self.dataset.prev_event[batch_index], dtype=torch.long, device=self.device),
                "duration": torch.as_tensor(self.dataset.duration[batch_index], dtype=torch.float32, device=self.device),
                "duration_model": torch.as_tensor(self.dataset.duration_model[batch_index], dtype=torch.float32, device=self.device),
                "duration_bucket": torch.as_tensor(self.dataset.duration_bucket[batch_index], dtype=torch.long, device=self.device),
                "ctn_model": torch.as_tensor(self.dataset.ctn_model[batch_index], dtype=torch.float32, device=self.device),
                "raw_sparse": torch.as_tensor(self.dataset.raw_sparse[batch_index], dtype=torch.long, device=self.device),
                "raw_seq": torch.as_tensor(self.dataset.raw_seq[batch_index], dtype=torch.long, device=self.device),
                "raw_seq_mask": torch.as_tensor(self.dataset.raw_seq_mask[batch_index], dtype=torch.float32, device=self.device),
                "content_category": torch.as_tensor(self.dataset.content_category[batch_index], dtype=torch.long, device=self.device),
                "event": torch.as_tensor(self.dataset.event[batch_index], dtype=torch.long, device=self.device),
                "session_key": torch.as_tensor(self.dataset.session_key[batch_index], dtype=torch.long, device=self.device),
            }
            label = torch.as_tensor(self.dataset.play_time[batch_index], dtype=torch.float32, device=self.device)
            yield features, label

        logger.info("Baseline batch iteration complete. rows=%d", row_count)


class DSIBaselineDataLoaders:
    def __init__(
        self,
        processed_dir: str | Path,
        device: str | torch.device,
        batch_size: int,
        max_train_rows: int | None = None,
        max_eval_rows: int | None = None,
        num_workers: int = 0,
        split_prefix: str = "standard",
        include_duration_bucket: bool = False,
        duration_bucket_count: int = 30,
        include_raw_features: bool = False,
    ) -> None:
        logger.info(
            "Building the baseline data loader. processed_dir=%s batch_size=%d split_prefix=%s include_raw=%s",
            processed_dir,
            int(batch_size),
            split_prefix,
            include_raw_features,
        )

        self.processed_dir = Path(processed_dir)
        self.device = torch.device(device)
        self.split_prefix = str(split_prefix)
        self.include_raw_features = bool(include_raw_features)
        manifest_path = self.processed_dir / "manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)

        content_meta = self.manifest.get("content_features", {})
        raw_meta = self.manifest.get("raw_features", {}) if self.include_raw_features else {}
        ctn_dim = (
            int(self.manifest.get("num_numeric_features", 0))
            + int(self.manifest.get("num_session_numeric_features", 0))
            + 1
            + int(raw_meta.get("num_raw_numeric_features", 0))
            + int(content_meta.get("num_content_numeric_features", 0))
        )

        train = DSIBaselineDataset(self.processed_dir / f"{self.split_prefix}_train.npz", max_train_rows, self.include_raw_features)
        val = DSIBaselineDataset(self.processed_dir / f"{self.split_prefix}_val.npz", max_eval_rows, self.include_raw_features)
        test = DSIBaselineDataset(self.processed_dir / f"{self.split_prefix}_test.npz", max_eval_rows, self.include_raw_features)
        logger.info("Configuring baseline model-side continuous-feature normalization.")
        self.model_continuous_scale = max(float(np.nanmax(train.play_time)), 1e-6)
        for dataset in (train, val, test):
            dataset.set_model_continuous_scale(self.model_continuous_scale)
        logger.info("Baseline model-side continuous-feature normalization configured. scale=%.6f", self.model_continuous_scale)

        self.duration_bucket_edges = np.zeros((0,), dtype=np.float32)
        self.duration_bucket_count = 1
        if include_duration_bucket:
            logger.info("Configuring duration_bucket input. requested_buckets=%d", int(duration_bucket_count))
            self.duration_bucket_edges = duration_quantile_edges(train.duration, int(duration_bucket_count))
            self.duration_bucket_count = int(self.duration_bucket_edges.size + 1)
            for dataset in (train, val, test):
                dataset.duration_bucket = np.searchsorted(
                    self.duration_bucket_edges,
                    dataset.duration.reshape(-1),
                    side="left",
                ).astype(np.int64)
            logger.info("Duration_bucket input configured. actual_buckets=%d", self.duration_bucket_count)

        self.description = [
            ("user_id", int(self.manifest["num_users"]), "spr"),
            ("video_id", int(self.manifest["num_items"]), "spr"),
            ("tab", int(self.manifest["num_tabs"]), "spr"),
            ("prev_event", int(self.manifest.get("unknown_prev_event", 5)) + 1, "spr"),
            ("raw_sparse", [int(value) for value in raw_meta.get("raw_sparse_cardinalities", [])], "spr_vec"),
            ("content_category", [int(value) for value in content_meta.get("content_category_cardinalities", [])], "spr_vec"),
            ("raw_seq", int(raw_meta.get("raw_sequence_cardinality", 0)), "seq"),
            ("duration_model", 1, "ctn"),
            ("ctn_model", ctn_dim, "ctn_vec"),
            ("play_time", 1, "label"),
        ]
        if include_duration_bucket:
            self.description.insert(4, ("duration_bucket", int(self.duration_bucket_count), "spr"))

        self.dataloaders = {
            "train": DSIBaselineBatchLoader(
                train,
                self.device,
                batch_size=batch_size,
                shuffle=True,
            ),
            "val": DSIBaselineBatchLoader(
                val,
                self.device,
                batch_size=batch_size,
                shuffle=False,
            ),
            "test": DSIBaselineBatchLoader(
                test,
                self.device,
                batch_size=batch_size,
                shuffle=False,
            ),
        }
        logger.info("Baseline data loader built. processed_dir=%s include_raw=%s", processed_dir, self.include_raw_features)

    def _collate(self, batch: list[tuple[dict[str, torch.Tensor], torch.Tensor]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:

        features_list, labels = zip(*batch)

        features = {
            key: torch.stack([item[key] for item in features_list]).to(self.device)
            for key in features_list[0]
        }
        labels_tensor = torch.stack(labels).to(self.device)
        return features, labels_tensor

    def __getitem__(self, split: str) -> DataLoader:
        return self.dataloaders[split]

    def keys(self) -> Iterator[str]:
        return iter(self.dataloaders.keys())
