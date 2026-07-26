#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dsi.commands import load_config as load_dsi_config
from dsi.utils.bridge import portableize_serialized_paths, resolve_path
from dsi.utils.logging_utils import setup_logging

from dsi.decision.features import _cached_feature_matrices, _play_time, _sample_train_indices
from dsi.decision.readout import _run_feature_set, _seed_offset


def _set_seed(seed: int, logger) -> None:
    logger.info("Setting random seeds. seed=%s", int(seed))

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    logger.info("Random seeds set. seed=%s", int(seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--feature-source", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--train-rows", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--feature-batch-size", type=int, default=8192)
    parser.add_argument("--scorers", nargs="+", choices=["mlp", "dcn"], default=["mlp", "dcn"])
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=["raw", "raw_padded", "posterior", "raw_equivalent", "posterior_equivalent", "fused"],
        default=["posterior"],
    )
    parser.add_argument("--posterior-summary-protocol", choices=["auto", "compact_27", "plain_time_27", "scalar_2", "full_time_69", "engineering", "hidden_only"], default="auto")
    parser.add_argument("--rank-target", choices=["long", "overplay"], default="overplay")
    parser.add_argument("--validation-select", choices=["mae", "xauc", "rank_auc"], default="rank_auc")
    parser.add_argument("--rank-loss-weight", type=float, default=0.5)
    parser.add_argument("--ordinal-bins", type=int, default=30)
    parser.add_argument("--ordinal-beta", type=float, default=3.0)
    parser.add_argument("--ordinal-restore-weight", type=float, default=30.0)
    parser.add_argument("--ordinal-ce-weight", type=float, default=0.0)
    parser.add_argument("--ordinal-prior-weight", type=float, default=0.0)
    parser.add_argument("--decision-id-features", action="store_true")
    parser.add_argument("--decision-id-embedding-dim", type=int, default=16)
    parser.add_argument("--decision-id-mode", choices=["append", "input"], default="append")
    parser.add_argument("--scorer-hidden-dim", type=int, default=128)
    parser.add_argument("--scorer-hidden-layers", type=int, default=2)
    parser.add_argument("--scorer-cross-layers", type=int, default=2)
    parser.add_argument("--scorer-dropout", type=float, default=0.1)
    parser.add_argument("--scorer-epochs", type=int, default=5)
    parser.add_argument("--scorer-lr", type=float, default=0.01)
    parser.add_argument("--scorer-weight-decay", type=float, default=0.0001)
    parser.add_argument("--scorer-clip-grad-norm", type=float, default=10.0)
    parser.add_argument("--topks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--refresh-feature-cache", action="store_true")
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--save-predictions", default=None)
    parser.add_argument("--fast-eval", action="store_true")
    parser.add_argument("--fast-eval-rows", type=int, default=200000)
    parser.add_argument("--feature-pad-dim", type=int, default=None)
    args = parser.parse_args()

    log_path = None if args.log_path is None else str(resolve_path(args.log_path))
    logger = setup_logging(log_path)
    logger.info("Running DSI posterior decision readout. seed=%s", int(args.seed))
    _set_seed(int(args.seed), logger)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    metadata, feature_sets, payload = _cached_feature_matrices(args, device, logger)
    requested_feature_sets = [str(value) for value in args.feature_sets]
    feature_sets = {name: feature_sets[name] for name in requested_feature_sets if name in feature_sets}
    if not feature_sets:
        raise ValueError(f"No requested feature sets exist: {requested_feature_sets}")

    logger.info("Aligning DSI readout input width. target=%s", args.feature_pad_dim)

    if args.feature_pad_dim is not None:
        target_dim = int(args.feature_pad_dim)
        for feature_name, matrices in feature_sets.items():
            for split, matrix in matrices.items():
                current_dim = int(matrix.shape[1])
                if target_dim < current_dim:
                    raise ValueError(f"--feature-pad-dim={target_dim} is smaller than {feature_name} dim {current_dim}")
                if target_dim > current_dim:
                    padding = np.zeros((matrix.shape[0], target_dim - current_dim), dtype=np.float32)
                    matrices[split] = np.concatenate([matrix, padding], axis=1).astype(np.float32, copy=False)
    logger.info("DSI readout input width aligned.")

    config, _ = load_dsi_config(args.base_config)
    processed_dir = Path(config["data"]["processed_dir"]).expanduser().resolve()
    train_npz = np.load(processed_dir / "standard_train.npz", allow_pickle=False)
    val_npz = np.load(processed_dir / "standard_val.npz", allow_pickle=False)
    test_npz = np.load(processed_dir / "standard_test.npz", allow_pickle=False)
    train_rows = int(metadata["train_rows"])
    train_idx = _sample_train_indices(int(train_npz["watch_event"].shape[0]), train_rows, int(args.seed), logger)
    play_time = {
        "train": _play_time(train_npz, train_idx),
        "val": _play_time(val_npz, None),
        "test": _play_time(test_npz, None),
    }
    labels = payload["labels"]
    eval_arrays = payload["eval"]
    id_features = payload.get("id_features")
    if bool(args.decision_id_features):
        args.decision_sparse_cardinalities = [
            int(value)
            for value in metadata.get("decision_sparse_cardinalities", [])
        ]

    results: dict[str, Any] = {}
    for scorer_name in [str(value) for value in args.scorers]:
        results[scorer_name] = {}
        for feature_set_name, matrices in feature_sets.items():
            _set_seed(int(args.seed) + _seed_offset(scorer_name, feature_set_name), logger)
            results[scorer_name][feature_set_name] = _run_feature_set(
                scorer_name,
                feature_set_name,
                matrices,
                labels,
                play_time,
                id_features,
                eval_arrays,
                device,
                args,
                logger,
            )

    output = {
        "method": "coarsened_support_posterior_ordinal_decision",
        "seed": int(args.seed),
        "rank_target": str(args.rank_target),
        "validation_select": str(args.validation_select),
        "decision_readout": "ordinal_survival_integral",
        "metadata": metadata,
        "scorers": [str(value) for value in args.scorers],
        "feature_sets": [str(value) for value in feature_sets],
        "scorer_config": {
            "hidden_dim": int(args.scorer_hidden_dim),
            "hidden_layers": int(args.scorer_hidden_layers),
            "cross_layers": int(args.scorer_cross_layers),
            "dropout": float(args.scorer_dropout),
            "epochs": int(args.scorer_epochs),
            "lr": float(args.scorer_lr),
            "rank_loss_weight": float(args.rank_loss_weight),
            "ordinal_bins": int(args.ordinal_bins),
            "ordinal_beta": float(args.ordinal_beta),
            "ordinal_restore_weight": float(args.ordinal_restore_weight),
            "ordinal_ce_weight": float(args.ordinal_ce_weight),
            "ordinal_prior_weight": float(args.ordinal_prior_weight),
            "decision_id_features": bool(args.decision_id_features),
            "decision_id_embedding_dim": int(args.decision_id_embedding_dim),
            "decision_id_mode": str(args.decision_id_mode),
            "fast_eval": bool(args.fast_eval),
            "fast_eval_rows": int(args.fast_eval_rows),
            "feature_pad_dim": None if args.feature_pad_dim is None else int(args.feature_pad_dim),
        },
        "results": results,
    }
    serialized_output = portableize_serialized_paths(output)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized_output, handle, indent=2, ensure_ascii=False)
    logger.info("DSI posterior decision readout complete. output=%s", output_path)
    print(json.dumps(serialized_output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
