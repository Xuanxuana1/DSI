from __future__ import annotations

import json
import logging
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from baselines.adapter import build_adapter
from baselines.data import DSIBaselineDataLoaders
from baselines.training import build_baseline, evaluate, train_base
from dsi.decision.features import FeatureSource, _load_content_arrays, _model_features, _raw_features
from dsi.decision.readout import _run_feature_set
from dsi.training import train_model
from dsi.utils.config import dump_config, load_config
from dsi.utils.logging_utils import setup_logging
from dsi.utils.time_bins import build_time_bin_upper_bounds

logger = logging.getLogger(__name__)


def _split_arrays(rows: int, seed: int, upper: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    logger.info("开始构造合成split。rows=%d seed=%d", int(rows), int(seed))

    duration = rng.uniform(4.0, 30.0, size=rows).astype(np.float32)
    event = np.arange(rows, dtype=np.int64) % 4
    ratios = np.asarray([0.15, 0.60, 1.00, 1.35], dtype=np.float32)[event]
    play = (duration * ratios * rng.uniform(0.96, 1.04, size=rows)).astype(np.float32)
    play = np.maximum(play, 0.05).astype(np.float32)

    numeric = np.column_stack(
        [
            np.log1p(duration),
            rng.normal(size=rows),
            rng.normal(size=rows),
            rng.normal(size=rows),
            rng.normal(size=rows),
            np.sin(np.arange(rows) / 24.0),
            np.cos(np.arange(rows) / 24.0),
        ]
    ).astype(np.float32)
    session_numeric = numeric[:, 1:].astype(np.float32, copy=True)
    time_bin = np.searchsorted(upper, play, side="left").clip(0, len(upper) - 1).astype(np.int64)
    arrays = {
        "user_idx": (np.arange(rows, dtype=np.int64) % 12),
        "item_idx": (np.arange(rows, dtype=np.int64) % 18),
        "tab": np.zeros(rows, dtype=np.int64),
        "prev_event": ((event + 3) % 4).astype(np.int64),
        "numeric": numeric,
        "session_numeric": session_numeric,
        "duration_sec": duration,
        "event": event,
        "watch_event": event.copy(),
        "time_bin": time_bin,
        "engagement_mark": np.isin(event, np.asarray([2, 3])).astype(np.float32),
        "play_time_sec": play,
        "session_key": (np.arange(rows, dtype=np.int64) // 8),
        "raw_sparse": np.zeros((rows, 0), dtype=np.int64),
        "raw_numeric": np.column_stack([duration / 30.0, ratios]).astype(np.float32),
        "raw_seq": np.zeros((rows, 0), dtype=np.int64),
        "raw_seq_mask": np.zeros((rows, 0), dtype=np.float32),
    }
    logger.info("合成split构造完成。rows=%d", int(rows))
    return arrays


def _write_fixture(root: Path) -> Path:
    logger.info("开始写出smoke fixture。root=%s", root)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    upper = build_time_bin_upper_bounds(60, 1800, 1.5)

    counts = {"train": 128, "val": 64, "test": 64}
    for offset, (split, rows) in enumerate(counts.items()):
        np.savez(processed / f"standard_{split}.npz", **_split_arrays(rows, 100 + offset, upper))

    manifest = {
        "dataset": "DSI-Smoke",
        "processed_dir": str(processed),
        "event_names": ["early_skip", "mid_watch", "true_completion", "overplay"],
        "watch_event_names": ["early_skip", "mid_watch", "true_completion", "overplay"],
        "unknown_prev_event": 4,
        "time_bin_upper_bounds_sec": upper.tolist(),
        "num_time_bins": int(len(upper)),
        "num_users": 12,
        "num_items": 18,
        "num_tabs": 1,
        "num_numeric_features": 7,
        "num_session_numeric_features": 6,
        "content_features": {
            "enabled": False,
            "num_content_category_levels": 0,
            "content_category_cardinalities": [],
            "num_content_numeric_features": 0,
        },
        "raw_features": {
            "enabled": True,
            "raw_sparse_cardinalities": [],
            "num_raw_sparse_features": 0,
            "num_raw_numeric_features": 2,
            "raw_sequence_cardinality": 0,
            "raw_sequence_length": 0,
        },
        "split_counts": {f"standard_{key}": int(value) for key, value in counts.items()},
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("smoke fixture写出完成。processed=%s", processed)
    return processed


def _smoke_config(root: Path, processed: Path) -> dict:
    logger.info("开始构造smoke配置。")
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "datasets" / "kuairec.yaml")

    config["data"]["name"] = "DSI-Smoke"
    config["data"]["processed_dir"] = str(processed)
    config["outputs"]["root_dir"] = str(root / "runs")
    config["outputs"]["train_log"] = str(root / "train.log")
    config["model"]["embedding_dim"] = 8
    config["model"]["hidden_dim"] = 16
    config["training"].update({"seed": 2027, "device": "cpu", "batch_size": 32, "eval_batch_size": 64, "epochs": 1, "patience": 1, "num_workers": 0})
    config["evaluation"].update({"max_batches": 2, "cindex_max_pairs_per_bucket": 100})
    config["calibration"]["enabled"] = False
    config["debug"].update({"max_train_rows": None, "max_eval_rows": None})
    smoke_config_path = root / "smoke.yaml"
    config["_config_path"] = str(smoke_config_path)
    dump_config(config, smoke_config_path, portable_paths=False)
    logger.info("smoke配置构造完成。")
    return config


def _smoke_readout(config: dict, processed: Path, run_root: Path) -> dict:
    logger.info("开始smoke决策读出。")
    device = torch.device("cpu")
    source = FeatureSource("dsi", "obs_set_posterior", Path(config["_config_path"]), run_root)
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    content = _load_content_arrays(processed)

    from dsi.decision.features import _prepare_model

    source_config, model, source_manifest = _prepare_model(source, 2027, device, logger)
    splits = {}
    raw = {}
    labels = {"long": {}, "overplay": {}}
    play_time = {}
    test_eval = None
    for split in ("train", "val", "test"):
        with np.load(processed / f"standard_{split}.npz", allow_pickle=False) as npz:
            raw[split], session = _raw_features(npz, None)
            event = np.asarray(npz["watch_event"], dtype=np.int64)
            labels["long"][split] = np.isin(event, np.asarray([2, 3])).astype(np.float32)
            labels["overplay"][split] = (event == 3).astype(np.float32)
            play_time[split] = np.asarray(npz["play_time_sec"], dtype=np.float32)
            if split == "test":
                test_eval = {
                    "play_time": play_time[split].astype(np.float64),
                    "watch_event": event,
                    "duration_sec": np.asarray(npz["duration_sec"], dtype=np.float64),
                    "session_key": np.asarray(session, dtype=np.int64),
                    "long_label": labels["long"][split].astype(np.float64),
                    "overplay_label": labels["overplay"][split].astype(np.float64),
                }
        posterior = _model_features(source, source_config, model, source_manifest, f"standard_{split}", None, 64, device, content, "compact_27", logger)
        splits[split] = np.concatenate([raw[split], posterior], axis=1).astype(np.float32)

    from argparse import Namespace

    args = Namespace(
        seed=2027,
        scorer_hidden_dim=16,
        scorer_hidden_layers=1,
        scorer_cross_layers=1,
        scorer_dropout=0.0,
        scorer_epochs=1,
        scorer_lr=0.01,
        scorer_weight_decay=0.0001,
        scorer_clip_grad_norm=10.0,
        rank_target="overplay",
        validation_select="mae",
        rank_loss_weight=0.5,
        ordinal_bins=8,
        ordinal_beta=3.0,
        ordinal_restore_weight=5.0,
        ordinal_ce_weight=0.0,
        ordinal_prior_weight=0.0,
        batch_size=32,
        topks=[1, 3, 5],
        fast_eval=False,
        fast_eval_rows=200000,
        decision_id_features=False,
        decision_id_embedding_dim=8,
        decision_id_mode="append",
        decision_sparse_cardinalities=[],
    )
    result = _run_feature_set("dcn", "fused", splits, labels, play_time, None, test_eval, device, args, logger)
    logger.info("smoke决策读出完成。")
    return result


def _smoke_baseline(config: dict, processed: Path) -> dict:
    logger.info("开始smoke baseline。")
    device = torch.device("cpu")
    loaders = DSIBaselineDataLoaders(str(processed), device, 32, num_workers=0, include_raw_features=False)

    from argparse import Namespace

    args = Namespace(
        baseline="vr",
        embed_dim=8,
        mlp_dims=[16, 8],
        share_mlp_dims=[16, 8],
        dropout=0.0,
        egmn_components=4,
        egmn_dropout=0.0,
        tpm_bins=8,
        duration_buckets=5,
        d2q_quantile_bins=16,
        cwm_embed_dim=8,
        cwm_c_inv=40.0,
        cwm_sigma=2.0,
        cread_heads=8,
        cread_beta=50.0,
        cread_lambda_ce=1.0,
        cread_lambda_restore=1.0,
        cread_lambda_ord=0.00002,
        cread_share_mlp_dims=[16, 8],
        cread_output_dims=[8],
        epochs=1,
        lr=0.01,
        weight_decay=0.000001,
        clip_grad_norm=10.0,
    )
    model = build_baseline("vr", loaders.description, args).to(device)
    adapter = build_adapter("vr", model).to(device)
    train_base(adapter, loaders, args, device)
    result = evaluate(adapter, loaders, "test", device)
    logger.info("smoke baseline完成。")
    return result


def run_smoke() -> dict:
    global logger
    logger = setup_logging(None)
    logger.info("开始完整smoke链路。")

    with tempfile.TemporaryDirectory(prefix="dsi-smoke-") as temp:
        root = Path(temp)
        processed = _write_fixture(root)
        config = _smoke_config(root, processed)

        typed = train_model(deepcopy(config), "obs_set_posterior", logger)
        time_only_config = deepcopy(config)
        time_only = train_model(time_only_config, "time_only_posterior", logger)
        readout = _smoke_readout(config, processed, Path(config["outputs"]["root_dir"]))
        baseline = _smoke_baseline(config, processed)

        readout_test = readout
        readout_values = {
            "mae": readout_test["watch_time_error"]["mae_sec"],
            "xauc": readout_test["watch_time_ranking"]["xauc_global"],
            "c_ndcg@5": readout_test["rank_head"]["ndcg"]["long_label"]["ndcg@5"],
            "op_ndcg@5": readout_test["rank_head"]["ndcg"]["overplay_label"]["ndcg@5"],
        }
        required_baseline = {"mae", "xauc", "c_ndcg@5", "op_ndcg@5"}
        if not required_baseline.issubset(baseline):
            raise AssertionError(f"Missing baseline metrics: {sorted(required_baseline - set(baseline))}")
        if not Path(config["outputs"]["root_dir"]).joinpath("obs_set_posterior", "seed_2027", "best.pt").exists():
            raise AssertionError("Typed provider checkpoint was not created")
        if not Path(config["outputs"]["root_dir"]).joinpath("time_only_posterior", "seed_2027", "best.pt").exists():
            raise AssertionError("Time-only provider checkpoint was not created")

        result = {
            "typed_primary_nll": float(typed["test"]["primary_nll"]),
            "time_only_primary_nll": float(time_only["test"]["primary_nll"]),
            "readout": {key: float(value) for key, value in readout_values.items()},
            "baseline": {key: float(baseline[key]) for key in sorted(required_baseline)},
        }
    logger.info("完整smoke链路完成。result=%s", result)
    return result
