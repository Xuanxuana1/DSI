from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from dsi import DSI_MODEL
from dsi.commands import (
    dsi_feature_source,
    load_config,
    log_path,
    module_argv,
    report_path,
    run_module_main,
)
from dsi.utils.config import load_config as load_yaml
from dsi.utils.logging_utils import setup_logging
from preprocess.kuairec import prepare_kuairec
from preprocess.kuairand import prepare_kuairand
from preprocess.wechat21 import prepare_wechat21


def _paper_config() -> dict[str, Any]:
    return load_yaml(Path(__file__).resolve().parents[1] / "configs" / "experiments" / "paper.yaml")


def _dataset_kind(config: dict[str, Any]) -> str:
    data = config.get("data", {})
    if "user_action_file" in data:
        return "wechat21"
    if "matrix" in data:
        return "kuairec"
    return "kuairand"


def _resolve_train_rows(config: dict[str, Any], value: Any) -> int:
    if str(value).lower() != "full":
        return int(value)

    manifest_path = Path(config["data"]["processed_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return int(manifest["split_counts"]["standard_train"])


def _run_prepare(args: argparse.Namespace) -> None:
    config, _ = load_config(args.config)
    logger = setup_logging(config["outputs"]["prepare_log"])
    logger.info("开始数据预处理。dataset=%s", config["data"]["name"])

    kind = _dataset_kind(config)
    if kind == "kuairec":
        prepare_kuairec(config, logger)
    elif kind == "kuairand":
        prepare_kuairand(config, logger)
    else:
        prepare_wechat21(config, logger)
    logger.info("数据预处理完成。processed_dir=%s", config["data"]["processed_dir"])


def _run_train(args: argparse.Namespace) -> None:
    config, _ = load_config(args.config)
    logger = setup_logging(config["outputs"]["train_log"])
    logger.info("开始训练provider。variant=%s seed=%d", args.variant, int(args.seed))

    config["training"]["seed"] = int(args.seed)
    config["training"]["device"] = str(args.device)
    model_type = DSI_MODEL if args.variant == "typed" else "time_only_posterior"

    from dsi.training import train_model

    train_model(config, model_type, logger)
    logger.info("provider训练完成。variant=%s", args.variant)


def _run_readout(args: argparse.Namespace) -> None:
    config, config_path = load_config(args.config)
    protocol = "controlled_decision" if bool(getattr(args, "controlled", False)) or args.interface in {"scalar", "compact", "full", "plain"} else "main_decision"
    paper = _paper_config()[protocol]
    report_kind = "decision" if protocol == "main_decision" and args.interface == "fused" else f"decision_{args.interface}"
    output = report_path(config_path, report_kind, int(args.seed), None)
    run_log = log_path(config_path, report_kind, int(args.seed), None)
    logger = setup_logging(run_log)
    logger.info("开始决策读出。interface=%s seed=%d", args.interface, int(args.seed))

    interface_map = {
        "fused": ("auto", "fused"),
        "posterior": ("auto", "posterior"),
        "scalar": ("scalar_2", "fused"),
        "compact": ("compact_27", "fused"),
        "full": ("full_time_69", "fused"),
        "plain": ("plain_time_27", "fused"),
    }
    summary, feature_set = interface_map[args.interface]
    command = [
        "--base-config", str(config_path),
        "--feature-source", dsi_feature_source(config, config_path),
        "--output", str(output),
        "--seed", str(int(args.seed)),
        "--device", str(args.device),
        "--log-path", str(run_log),
        "--train-rows", str(_resolve_train_rows(config, paper["train_rows"])),
        "--batch-size", str(int(paper["batch_size"])),
        "--feature-batch-size", str(int(paper["feature_batch_size"])),
        "--scorers", str(paper["scorer"]),
        "--feature-sets", feature_set,
        "--posterior-summary-protocol", summary,
        "--rank-target", str(paper["rank_target"]),
        "--validation-select", str(paper["validation_select"]),
        "--rank-loss-weight", str(float(paper["rank_loss_weight"])),
        "--ordinal-bins", str(int(paper["ordinal_bins"])),
        "--ordinal-beta", str(float(paper["ordinal_beta"])),
        "--ordinal-restore-weight", str(float(paper["ordinal_restore_weight"])),
        "--ordinal-ce-weight", str(float(paper["ordinal_ce_weight"])),
        "--ordinal-prior-weight", str(float(paper["ordinal_prior_weight"])),
        "--scorer-hidden-dim", str(int(paper["hidden_dim"])),
        "--scorer-hidden-layers", str(int(paper["hidden_layers"])),
        "--scorer-cross-layers", str(int(paper["cross_layers"])),
        "--scorer-dropout", str(float(paper["dropout"])),
        "--scorer-epochs", str(int(paper["epochs"])),
        "--scorer-lr", str(float(paper["learning_rate"])),
        "--scorer-weight-decay", str(float(paper["weight_decay"])),
    ]

    from dsi.decision.eval import main as readout_main

    run_module_main(readout_main, module_argv("dsi readout", command))
    logger.info("决策读出完成。output=%s", output)


def _run_baseline(args: argparse.Namespace) -> None:
    config, config_path = load_config(args.config)
    paper = _paper_config()["baseline"]
    output = report_path(config_path, f"baseline_{args.method}", int(args.seed), None)
    run_log = log_path(config_path, f"baseline_{args.method}", int(args.seed), None)
    setup_logging(run_log).info("开始训练baseline。method=%s seed=%d", args.method, int(args.seed))

    command = [
        "--config", str(config_path),
        "--baseline", str(args.method),
        "--seed", str(int(args.seed)),
        "--device", str(args.device),
        "--output", str(output),
        "--log-path", str(run_log),
        "--batch-size", str(int(paper["batch_size"])),
        "--epochs", str(int(paper["epochs"])),
        "--feature-protocol", str(paper["feature_protocol"]),
    ]
    from baselines.training import main as baseline_main

    run_module_main(baseline_main, module_argv("dsi baseline", command))


def _run_matched(args: argparse.Namespace) -> None:
    config, config_path = load_config(args.config)
    paper = _paper_config()["matched_readout"]
    output = report_path(config_path, f"matched_{args.method}", int(args.seed), None)
    run_log = log_path(config_path, f"matched_{args.method}", int(args.seed), None)
    setup_logging(run_log).info("开始匹配读出。method=%s seed=%d", args.method, int(args.seed))

    command = [
        "--config", str(config_path),
        "--baselines", str(args.method),
        "--seed", str(int(args.seed)),
        "--device", str(args.device),
        "--output", str(output),
        "--log-path", str(run_log),
        "--train-rows", str(int(paper["train_rows"])),
        "--baseline-batch-size", str(int(paper["baseline_batch_size"])),
        "--batch-size", str(int(paper["readout_batch_size"])),
        "--feature-batch-size", str(int(paper["feature_batch_size"])),
        "--scorer", str(paper["scorer"]),
        "--representation-protocol", str(paper["representation_protocol"]),
        "--rank-target", str(paper["rank_target"]),
        "--validation-select", str(paper["validation_select"]),
        "--rank-loss-weight", str(float(paper["rank_loss_weight"])),
        "--ordinal-bins", str(int(paper["ordinal_bins"])),
        "--ordinal-beta", str(float(paper["ordinal_beta"])),
        "--ordinal-restore-weight", str(float(paper["ordinal_restore_weight"])),
        "--scorer-epochs", str(int(paper["epochs"])),
        "--scorer-lr", str(float(paper["learning_rate"])),
    ]
    from baselines.readout_probe import main as matched_main

    run_module_main(matched_main, module_argv("dsi matched", command))


def _run_ablation(args: argparse.Namespace) -> None:
    config, config_path = load_config(args.config)
    if str(args.variant) == "time-only":
        protocol = "time_only_decision"
    elif str(args.variant) == "plain-summary":
        protocol = "controlled_decision"
    else:
        protocol = "main_decision"
    paper = _paper_config()[protocol]
    variant = str(args.variant)
    run_config = copy.deepcopy(config)
    run_config["training"]["seed"] = int(args.seed)
    run_config["training"]["device"] = str(args.device)
    model_type = DSI_MODEL
    summary = "auto"
    feature_set = "fused"

    if variant == "nll-only":
        run_config["loss"]["lambda_obs_set_restore"] = 0.0
    elif variant == "time-only":
        model_type = "time_only_posterior"
    elif variant == "plain-summary":
        model_type = "time_only_posterior"
        summary = "plain_time_27"
    elif variant == "raw":
        feature_set = "raw"
    elif variant == "posterior":
        feature_set = "posterior"
    elif variant == "value-only":
        paper = copy.deepcopy(paper)
        paper["rank_loss_weight"] = 0.0
        paper["validation_select"] = "mae"

    needs_provider = variant in {"nll-only", "time-only", "plain-summary"}
    if needs_provider:
        base_run_root = Path(config["outputs"]["root_dir"])
        run_config["outputs"]["root_dir"] = str(
            base_run_root.parent / f"{base_run_root.name}_{variant}"
        )
        run_config["outputs"]["train_log"] = str(log_path(config_path, f"ablation_{variant}_provider", int(args.seed), None))
        provider_logger = setup_logging(run_config["outputs"]["train_log"])
        provider_logger.info("开始机制消融provider。variant=%s", variant)
        from dsi.training import train_model

        train_model(run_config, model_type, provider_logger)
        provider_logger.info("机制消融provider完成。variant=%s", variant)

    from dsi.utils.config import dump_config

    ablation_config = Path(__file__).resolve().parents[1] / "artifacts" / "configs" / f"{config_path.stem}_{variant}.yaml"
    dump_config(run_config, ablation_config)
    output = report_path(config_path, f"ablation_{variant}", int(args.seed), None)
    run_log = log_path(config_path, f"ablation_{variant}", int(args.seed), None)
    source_config = ablation_config if needs_provider else config_path
    source = None if variant == "raw" else f"dsi:{model_type}:{source_config}:{run_config['outputs']['root_dir']}"
    command = [
        "--base-config", str(config_path),
        "--output", str(output),
        "--seed", str(int(args.seed)),
        "--device", str(args.device),
        "--log-path", str(run_log),
        "--train-rows", str(_resolve_train_rows(config, paper["train_rows"])),
        "--batch-size", str(int(paper["batch_size"])),
        "--feature-batch-size", str(int(paper["feature_batch_size"])),
        "--scorers", str(paper["scorer"]),
        "--feature-sets", feature_set,
        "--posterior-summary-protocol", summary,
        "--rank-target", str(paper["rank_target"]),
        "--validation-select", str(paper["validation_select"]),
        "--rank-loss-weight", str(float(paper["rank_loss_weight"])),
        "--ordinal-bins", str(int(paper["ordinal_bins"])),
        "--ordinal-beta", str(float(paper["ordinal_beta"])),
        "--ordinal-restore-weight", str(float(paper["ordinal_restore_weight"])),
        "--ordinal-ce-weight", str(float(paper["ordinal_ce_weight"])),
        "--ordinal-prior-weight", str(float(paper["ordinal_prior_weight"])),
        "--scorer-hidden-dim", str(int(paper["hidden_dim"])),
        "--scorer-hidden-layers", str(int(paper["hidden_layers"])),
        "--scorer-cross-layers", str(int(paper["cross_layers"])),
        "--scorer-dropout", str(float(paper["dropout"])),
        "--scorer-epochs", str(int(paper["epochs"])),
        "--scorer-lr", str(float(paper["learning_rate"])),
    ]
    if source is not None:
        command.extend(["--feature-source", source])
    from dsi.decision.eval import main as ablation_main

    run_module_main(ablation_main, module_argv("dsi ablation", command))


def _run_boundary(args: argparse.Namespace) -> None:
    config, config_path = load_config(args.config)
    boundary = str(args.boundary)
    value = float(args.value)
    run_config = copy.deepcopy(config)
    run_config["training"]["seed"] = int(args.seed)
    run_config["training"]["device"] = str(args.device)
    if boundary == "alpha":
        run_config["event_taxonomy"]["alpha"] = value
    elif boundary == "rho":
        run_config["event_taxonomy"]["mid_completion_cut"] = value
    else:
        run_config["event_taxonomy"]["beta"] = value
        run_config["event_taxonomy"]["overplay_band_beta"] = value

    slug = str(value).replace(".", "p")
    name = f"{config_path.stem}_{boundary}_{slug}"
    root = Path(__file__).resolve().parents[1]
    run_config["data"]["processed_dir"] = str(root / "artifacts" / "boundary" / "processed" / name)
    run_config["outputs"]["root_dir"] = str(root / "artifacts" / "boundary" / "runs" / name)
    run_config["outputs"]["prepare_log"] = str(root / "artifacts" / "boundary" / "logs" / f"{name}_prepare.log")
    run_config["outputs"]["train_log"] = str(root / "artifacts" / "boundary" / "logs" / f"{name}_train.log")
    from dsi.utils.config import dump_config

    boundary_config = root / "artifacts" / "configs" / f"{name}.yaml"
    dump_config(run_config, boundary_config)

    prepare_args = argparse.Namespace(config=str(boundary_config))
    _run_prepare(prepare_args)
    train_args = argparse.Namespace(config=str(boundary_config), seed=int(args.seed), device=str(args.device), variant="typed")
    _run_train(train_args)
    readout_args = argparse.Namespace(config=str(boundary_config), seed=int(args.seed), device=str(args.device), interface="fused", controlled=False)
    _run_readout(readout_args)


def _run_transfer(args: argparse.Namespace) -> None:
    config, config_path = load_config(args.config)
    paper = _paper_config()["transfer"]
    source = str(args.source).replace("-", "_")
    output = report_path(config_path, f"transfer_{source}_{args.task}", int(args.seed), None)
    run_log = log_path(config_path, f"transfer_{source}_{args.task}", int(args.seed), None)
    dsi_report = report_path(config_path, "decision", int(args.seed), None)
    setup_logging(run_log).info("开始迁移probe。source=%s task=%s", source, args.task)

    command = [
        "--config", str(config_path),
        "--source", source,
        "--seed", str(int(args.seed)),
        "--device", str(args.device),
        "--output", str(output),
        "--log-path", str(run_log),
        "--dsi-report", str(dsi_report),
        "--input-dim", str(int(paper["input_dim"])),
        "--fractions", *[str(float(value)) for value in paper["fractions"]],
        "--probe-steps", str(int(paper["probe_steps"])),
        "--eval-every-steps", str(int(paper["eval_every_steps"])),
        "--lr", str(float(paper["learning_rate"])),
        "--batch-size", str(int(paper["batch_size"])),
        "--feature-batch-size", str(int(paper["feature_batch_size"])),
        "--head", str(paper["head"]),
        "--quantile", str(float(paper["quantile"])),
        "--ratio-threshold", str(float(paper["ratio_threshold"])),
    ]
    if args.task == "engagement":
        command.extend(["--engagement-probe", "--engagement-only"])
    from baselines.transfer_probe import main as transfer_main

    run_module_main(transfer_main, module_argv("dsi transfer", command))


def _run_smoke(_: argparse.Namespace) -> None:
    logger = setup_logging(None)
    logger.info("开始smoke test。")

    from tests.smoke import run_smoke

    run_smoke()
    logger.info("smoke test完成。")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DSI paper reproduction code")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="preprocess a supported dataset")
    prepare.add_argument("--config", required=True)
    prepare.set_defaults(func=_run_prepare)

    train = commands.add_parser("train", help="train the distribution provider")
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int, default=2027)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--variant", choices=["typed", "time-only"], default="typed")
    train.set_defaults(func=_run_train)

    readout = commands.add_parser("readout", help="train the decision readout")
    readout.add_argument("--config", required=True)
    readout.add_argument("--seed", type=int, default=2027)
    readout.add_argument("--device", default="cuda:0")
    readout.add_argument("--interface", choices=["fused", "posterior", "scalar", "compact", "full", "plain"], default="fused")
    readout.set_defaults(func=_run_readout)

    baseline = commands.add_parser("baseline", help="train a main-table baseline")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--seed", type=int, default=2027)
    baseline.add_argument("--device", default="cuda:0")
    baseline.add_argument("--method", choices=["vr", "wlr", "cwm", "difl", "tpm", "cread", "egmn", "d2q", "gr"], required=True)
    baseline.set_defaults(func=_run_baseline)

    matched = commands.add_parser("matched", help="attach the shared readout to a baseline")
    matched.add_argument("--config", required=True)
    matched.add_argument("--seed", type=int, default=2027)
    matched.add_argument("--device", default="cuda:0")
    matched.add_argument("--method", choices=["d2q", "tpm", "egmn"], required=True)
    matched.set_defaults(func=_run_matched)

    ablation = commands.add_parser("ablation", help="run a mechanism ablation")
    ablation.add_argument("--config", required=True)
    ablation.add_argument("--seed", type=int, default=2027)
    ablation.add_argument("--device", default="cuda:0")
    ablation.add_argument("--variant", choices=["raw", "posterior", "value-only", "nll-only", "time-only", "plain-summary"], required=True)
    ablation.set_defaults(func=_run_ablation)

    boundary = commands.add_parser("boundary", help="run one event-boundary sensitivity point")
    boundary.add_argument("--config", required=True)
    boundary.add_argument("--seed", type=int, default=2027)
    boundary.add_argument("--device", default="cuda:0")
    boundary.add_argument("--boundary", choices=["alpha", "rho", "beta"], required=True)
    boundary.add_argument("--value", type=float, required=True)
    boundary.set_defaults(func=_run_boundary)

    transfer = commands.add_parser("transfer", help="run a frozen-representation probe")
    transfer.add_argument("--config", required=True)
    transfer.add_argument("--seed", type=int, default=2027)
    transfer.add_argument("--device", default="cuda:0")
    transfer.add_argument("--source", choices=["raw", "random-dsi", "dsi", "d2q", "tpm", "cread", "egmn"], required=True)
    transfer.add_argument("--task", choices=["watch", "engagement"], default="watch")
    transfer.set_defaults(func=_run_transfer)

    smoke = commands.add_parser("smoke", help="run the self-contained smoke test")
    smoke.set_defaults(func=_run_smoke)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
