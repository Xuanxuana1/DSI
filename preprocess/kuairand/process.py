from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsi.utils.bridge import portableize_serialized_paths
from dsi.utils.time_bins import assign_time_bins, build_time_bin_upper_bounds

LOG_COLUMNS = [
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "profile_stay_time",
    "comment_stay_time",
    "is_profile_enter",
    "is_rand",
    "tab",
]
KUAI_RAND_LOG_SPARSE_COLUMNS = ["date", "is_rand"]
KUAI_RAND_LOG_NUMERIC_COLUMNS: list[str] = []
KUAI_RAND_USER_SPARSE_COLUMNS = [
    "user_active_degree",
    "is_lowactive_period",
    "is_live_streamer",
    "is_video_author",
    "follow_user_num_range",
    "fans_user_num_range",
    "friend_user_num_range",
    "register_days_range",
    *[f"onehot_feat{idx}" for idx in range(18)],
]
KUAI_RAND_USER_NUMERIC_COLUMNS = [
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
]
KUAI_RAND_VIDEO_BASIC_SPARSE_COLUMNS = [
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "music_id",
    "music_type",
]
KUAI_RAND_VIDEO_BASIC_NUMERIC_COLUMNS = [
    "video_duration_basic",
    "server_width",
    "server_height",
]
KUAI_RAND_VIDEO_STATISTIC_NUMERIC_COLUMNS = [
    "counts",
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration_stat",
    "complete_play_cnt",
    "complete_play_user_num",
    "valid_play_cnt",
    "valid_play_user_num",
    "long_time_play_cnt",
    "long_time_play_user_num",
    "short_time_play_cnt",
    "short_time_play_user_num",
    "play_progress",
    "comment_stay_duration",
    "like_cnt",
    "like_user_num",
    "click_like_cnt",
    "double_click_cnt",
    "cancel_like_cnt",
    "cancel_like_user_num",
    "comment_cnt",
    "comment_user_num",
    "direct_comment_cnt",
    "reply_comment_cnt",
    "delete_comment_cnt",
    "delete_comment_user_num",
    "comment_like_cnt",
    "comment_like_user_num",
    "follow_cnt",
    "follow_user_num_stat",
    "cancel_follow_cnt",
    "cancel_follow_user_num",
    "share_cnt",
    "share_user_num",
    "download_cnt",
    "download_user_num",
    "report_cnt",
    "report_user_num",
    "reduce_similar_cnt",
    "reduce_similar_user_num",
    "collect_cnt",
    "collect_user_num",
    "cancel_collect_cnt",
    "cancel_collect_user_num",
    "direct_comment_user_num",
    "reply_comment_user_num",
    "share_all_cnt",
    "share_all_user_num",
    "outsite_share_all_cnt",
]
RAW_SPARSE_PREFIX = "__raw_sparse_"
RAW_NUMERIC_PREFIX = "__raw_numeric_"
RAW_SEQUENCE_PREFIX = "__raw_seq_"
RAW_SEQUENCE_MASK_PREFIX = "__raw_seq_mask_"

WATCH_EVENT_NAMES = ["early_skip", "mid_watch", "true_completion", "overplay_replay"]
EVENT_NAMES = list(WATCH_EVENT_NAMES)
UNKNOWN_PREV_EVENT = len(EVENT_NAMES)
CATEGORY_ID_COLUMNS = [
    "first_level_category_id",
    "second_level_category_id",
    "third_level_category_id",
    "fourth_level_category_id",
]
CATEGORY_PROB_COLUMNS = [
    "first_level_category_prob",
    "second_level_category_prob",
    "third_level_category_prob",
    "fourth_level_category_prob",
]
CONTENT_NUMERIC_NAMES = [
    "first_level_category_prob",
    "second_level_category_prob",
    "third_level_category_prob",
    "fourth_level_category_prob",
]
EXPOSURE_NUMERIC_NAMES = [
    "log_duration",
    "session_pos",
    "elapsed_session_sec",
    "prev_gap_sec",
    "prev_play_sec",
    "hour_sin",
    "hour_cos",
]
SESSION_NUMERIC_NAMES = [
    "session_pos",
    "elapsed_session_sec",
    "prev_gap_sec",
    "prev_play_sec",
    "hour_sin",
    "hour_cos",
]


def _watch_event_names(config: dict[str, Any]) -> list[str]:

    configured = config.get("event_taxonomy", {}).get("watch_event_names")
    if configured:
        return [str(name) for name in configured]

    if "mid_completion_cut" in config.get("event_taxonomy", {}):
        return ["early_skip", "mid_watch", "true_completion", "overplay_replay"]

    return list(WATCH_EVENT_NAMES)


def _event_names(config: dict[str, Any]) -> list[str]:

    return _watch_event_names(config)


def _unknown_prev_event(config: dict[str, Any]) -> int:

    return len(_event_names(config))


def _read_log(path: Path, nrows: int | None, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=LOG_COLUMNS, nrows=nrows)
    df["source"] = source
    return df


def _log_files(data_dir: Path, stem: str) -> list[Path]:

    single = data_dir / f"{stem}.csv"
    if single.exists():
        return [single]

    parts = sorted(data_dir.glob(f"{stem}_part*.csv"))
    if not parts:
        raise FileNotFoundError(f"No KuaiRand log files found for stem: {stem}")
    return parts


def _load_logs(data_dir: Path, version_suffix: str, max_rows: dict[str, int | None]) -> tuple[pd.DataFrame, pd.DataFrame]:
    standard_files = [
        *_log_files(data_dir, f"log_standard_4_08_to_4_21_{version_suffix}"),
        *_log_files(data_dir, f"log_standard_4_22_to_5_08_{version_suffix}"),
    ]
    random_file = data_dir / f"log_random_4_22_to_5_08_{version_suffix}.csv"

    rows_per_standard_file = max_rows.get("standard_per_file")
    standard_parts = [
        _read_log(path, rows_per_standard_file, "standard")
        for path in standard_files
    ]
    standard = pd.concat(standard_parts, ignore_index=True)
    random = _read_log(random_file, max_rows.get("random"), "random")
    return standard, random


def _sanitize_duration(df: pd.DataFrame, policy: str, min_duration_ms: int) -> dict[str, int]:
    invalid = df["duration_ms"] <= 0
    audit = {
        "rows_before_duration_policy": int(len(df)),
        "invalid_duration_rows": int(invalid.sum()),
        "invalid_duration_policy": policy,
    }
    if policy == "fail" and invalid.any():
        raise ValueError(f"Found {audit['invalid_duration_rows']} rows with duration_ms <= 0")
    if policy == "drop":
        df.drop(index=df.index[invalid], inplace=True)
    elif policy == "clip":
        df.loc[invalid, "duration_ms"] = min_duration_ms
    elif policy != "keep":
        raise ValueError(f"Unknown invalid_duration_policy: {policy}")
    df["duration_ms_safe"] = df["duration_ms"].clip(lower=min_duration_ms)
    audit["rows_after_duration_policy"] = int(len(df))
    return audit


def _construct_events(
    df: pd.DataFrame,
    config: dict[str, Any],
    upper_bounds: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    taxonomy = config["event_taxonomy"]
    gamma_ms = int(taxonomy["gamma_minutes"] * 60_000)
    alpha = float(taxonomy["alpha"])
    beta = float(taxonomy["beta"])
    mid_completion_cut = taxonomy.get("mid_completion_cut")
    event_names = _event_names(config)
    watch_event_names = _watch_event_names(config)
    unknown_prev_event = _unknown_prev_event(config)
    min_duration_ms = int(config["preprocessing"]["min_duration_ms"])
    duration_audit = _sanitize_duration(df, config["preprocessing"]["invalid_duration_policy"], min_duration_ms)

    df.sort_values(["user_id", "time_ms", "video_id"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    grouped = df.groupby("user_id", sort=False)
    prev_time = grouped["time_ms"].shift(1)
    next_time = grouped["time_ms"].shift(-1)
    gap_ms = df["time_ms"] - prev_time
    next_gap_ms = next_time - df["time_ms"]

    new_session = prev_time.isna() | (gap_ms > gamma_ms)
    df["session_ordinal"] = new_session.groupby(df["user_id"], sort=False).cumsum().astype(np.int64) - 1
    session_index = pd.MultiIndex.from_arrays([df["user_id"].to_numpy(), df["session_ordinal"].to_numpy()])
    df["session_key"] = pd.factorize(session_index, sort=False)[0]
    session_group = df.groupby("session_key", sort=False)

    df["session_pos"] = session_group.cumcount().astype(np.int32)
    df["elapsed_session_sec"] = ((df["time_ms"] - session_group["time_ms"].transform("first")) / 1000.0).astype(np.float32)
    gap_sec = (gap_ms.fillna(0).clip(lower=0, upper=gamma_ms) / 1000.0).astype(np.float32)
    df["prev_gap_sec"] = np.where(new_session.to_numpy(), 0.0, gap_sec).astype(np.float32)
    df["prev_play_sec"] = (session_group["play_time_ms"].shift(1).fillna(0).clip(lower=0) / 1000.0).astype(np.float32)

    ratio = df["play_time_ms"].clip(lower=0).to_numpy(np.float64) / df["duration_ms_safe"].to_numpy(np.float64)
    watch_event = np.zeros(len(df), dtype=np.int64)
    if mid_completion_cut is None:
        watch_event[(ratio >= alpha) & (ratio <= beta)] = 1
        watch_event[ratio > beta] = 2
    else:
        mid_cut = float(mid_completion_cut)
        if not alpha < mid_cut < beta:
            raise ValueError("event_taxonomy.mid_completion_cut must satisfy alpha < cut < beta")
        watch_event[(ratio >= alpha) & (ratio < mid_cut)] = 1
        watch_event[(ratio >= mid_cut) & (ratio <= beta)] = 2
        watch_event[ratio > beta] = 3

    is_session_end = next_time.isna() | (next_gap_ms > gamma_ms)
    df["event"] = watch_event.astype(np.int64)
    df["watch_event"] = watch_event.astype(np.int64)
    df["is_session_end"] = is_session_end.astype(np.int8)

    event_time_seconds = df["play_time_ms"].clip(lower=0).to_numpy(np.float64) / 1000.0
    df["time_bin"] = assign_time_bins(event_time_seconds, upper_bounds)
    df["engagement_mark"] = (
        df[["is_like", "is_follow", "is_comment", "is_forward"]].max(axis=1).astype(np.int8)
    )

    prev_event = session_group["event"].shift(1).fillna(unknown_prev_event)
    df["prev_event"] = prev_event.astype(np.int64)
    hour = (df["hourmin"].clip(lower=0) // 100).clip(upper=23).to_numpy(np.float32)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    df["duration_sec"] = (df["duration_ms_safe"] / 1000.0).astype(np.float32)
    df["log_duration"] = np.log1p(df["duration_sec"]).astype(np.float32)
    df["play_time_sec"] = (df["play_time_ms"].clip(lower=0) / 1000.0).astype(np.float32)

    event_counts = df["event"].value_counts().reindex(range(len(event_names)), fill_value=0)
    watch_event_counts = df["watch_event"].value_counts().reindex(range(len(watch_event_names)), fill_value=0)
    audit = {
        **duration_audit,
        "sessions": int(df["session_key"].nunique()),
        "rows_after_event_construction": int(len(df)),
        "event_counts": {event_names[idx]: int(count) for idx, count in event_counts.items()},
        "watch_event_counts": {watch_event_names[idx]: int(count) for idx, count in watch_event_counts.items()},
        "session_end_count": int(df["is_session_end"].sum()),
        "watch_event_names": watch_event_names,
    }
    return df, audit


def _assign_splits(df: pd.DataFrame, ratios: list[float]) -> pd.Series:
    session_info = (
        df.groupby("session_key", sort=False)
        .agg(user_id=("user_id", "first"), session_start_ms=("time_ms", "min"))
        .reset_index()
        .sort_values(["user_id", "session_start_ms", "session_key"], kind="mergesort")
    )
    session_info["rank"] = session_info.groupby("user_id", sort=False).cumcount()
    session_info["count"] = session_info.groupby("user_id", sort=False)["session_key"].transform("count")
    frac = (session_info["rank"] + 0.5) / session_info["count"].clip(lower=1)
    train_cut, val_cut = ratios[0], ratios[0] + ratios[1]
    session_info["split"] = np.select(
        [frac < train_cut, frac < val_cut],
        ["train", "val"],
        default="test",
    )
    mapping = dict(zip(session_info["session_key"], session_info["split"]))
    return df["session_key"].map(mapping)


def _assign_row_shuffle_splits(df: pd.DataFrame, ratios: list[float], random_state: int = 1234) -> pd.Series:

    shuffled_index = df.sample(frac=1.0, random_state=int(random_state)).index.to_numpy()
    n_rows = int(len(shuffled_index))

    n_train = int(float(ratios[0]) * n_rows)
    n_val_end = int(float(ratios[0] + ratios[1]) * n_rows)

    split = pd.Series("test", index=df.index, dtype=object)
    split.loc[shuffled_index[:n_train]] = "train"
    split.loc[shuffled_index[n_train:n_val_end]] = "val"
    split.loc[shuffled_index[n_val_end:]] = "test"
    return split


def _assign_configured_splits(df: pd.DataFrame, config: dict[str, Any]) -> pd.Series:

    split_cfg = config.get("split", {})
    unit = str(split_cfg.get("unit", "row_shuffle_80_10_10"))
    ratios = [float(value) for value in split_cfg.get("ratios", [0.7, 0.1, 0.2])]

    if unit == "user_session_time_order":
        return _assign_splits(df, ratios)
    if unit in {"row_shuffle", "row_shuffle_80_10_10"}:
        random_state = int(split_cfg.get("random_state", 1234))
        return _assign_row_shuffle_splits(df, ratios, random_state)

    raise ValueError(f"Unsupported split.unit={unit}")


def _encode_ids(standard: pd.DataFrame, random: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    all_users = pd.concat([standard["user_id"], random["user_id"]], ignore_index=True)
    all_items = pd.concat([standard["video_id"], random["video_id"]], ignore_index=True)
    user_codes, user_uniques = pd.factorize(all_users, sort=True)
    item_codes, item_uniques = pd.factorize(all_items, sort=True)

    standard_len = len(standard)
    standard["user_idx"] = user_codes[:standard_len].astype(np.int64)
    random["user_idx"] = user_codes[standard_len:].astype(np.int64)
    standard["item_idx"] = item_codes[:standard_len].astype(np.int64)
    random["item_idx"] = item_codes[standard_len:].astype(np.int64)

    item_lookup = pd.DataFrame(
        {
            "video_id": item_uniques.astype(np.int64),
            "item_idx": np.arange(len(item_uniques), dtype=np.int64),
        }
    )

    return (
        {
            "num_users": int(len(user_uniques)),
            "num_items": int(len(item_uniques)),
        },
        item_lookup,
    )


def _normalize_sparse_value(series: pd.Series) -> pd.Series:

    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    unknown_values = {"", "nan", "NaN", "None", "<NA>", "UNKNOWN", "unknown"}
    return text.mask(text.isin(unknown_values), "")


def _encode_sparse_columns(frames: list[pd.DataFrame], columns: list[str], logger) -> tuple[list[str], list[int], dict[str, dict[str, int]]]:
    logger.info("开始编码KuaiRand原始稀疏特征。cols=%s", len(columns))

    raw_columns: list[str] = []
    cardinalities: list[int] = []
    audit: dict[str, dict[str, int]] = {}
    for idx, column in enumerate(columns):
        normalized_parts = [
            _normalize_sparse_value(frame[column] if column in frame else pd.Series([""] * len(frame)))
            for frame in frames
        ]
        combined = pd.concat(normalized_parts, ignore_index=True)
        tokens = sorted(token for token in combined.drop_duplicates().tolist() if token)
        mapping = {token: code + 1 for code, token in enumerate(tokens)}
        raw_column = f"{RAW_SPARSE_PREFIX}{idx}"
        for frame, normalized in zip(frames, normalized_parts):
            frame[raw_column] = normalized.map(mapping).fillna(0).astype(np.int64)
        raw_columns.append(raw_column)
        cardinalities.append(len(mapping) + 1)
        audit[column] = {
            "cardinality": int(len(mapping) + 1),
            "missing_rows": int((combined == "").sum()),
        }

    logger.info("KuaiRand原始稀疏特征编码完成。cols=%s", len(raw_columns))
    return raw_columns, cardinalities, audit


def _encode_numeric_columns(frames: list[pd.DataFrame], columns: list[str], logger) -> tuple[list[str], dict[str, dict[str, int]]]:
    logger.info("开始编码KuaiRand原始数值特征。cols=%s", len(columns))

    raw_columns: list[str] = []
    audit: dict[str, dict[str, int]] = {}
    for idx, column in enumerate(columns):
        raw_column = f"{RAW_NUMERIC_PREFIX}{idx}"
        missing_total = 0
        for frame in frames:
            raw_value = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series([np.nan] * len(frame))
            missing_total += int(raw_value.isna().sum())
            values = raw_value.fillna(0.0).astype(np.float32).to_numpy(copy=False)
            if column != "play_progress":
                values = np.log1p(np.clip(values, 0, None)).astype(np.float32, copy=False)
            frame[raw_column] = values
        raw_columns.append(raw_column)
        audit[column] = {"missing_rows": missing_total}

    logger.info("KuaiRand原始数值特征编码完成。cols=%s", len(raw_columns))
    return raw_columns, audit


def _parse_feature_sequence(value: object) -> list[str]:

    if value is None:
        return []
    text = str(value).strip()
    if not text or text in {"nan", "NaN", "None", "UNKNOWN"}:
        return []

    parsed: object
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if parsed is not None and not isinstance(parsed, str):
        value_text = str(parsed).strip()
        return [value_text] if value_text else []

    normalized = text.replace("|", ",").replace(";", ",")
    if "," in normalized:
        return [item.strip() for item in normalized.split(",") if item.strip()]
    return [item.strip() for item in normalized.split() if item.strip()]


def _load_video_statistics(data_dir: Path, version_suffix: str, logger) -> pd.DataFrame:
    logger.info("开始读取KuaiRand video statistic原始表。suffix=%s", version_suffix)

    files = sorted(data_dir.glob(f"video_features_statistic_{version_suffix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No video statistic files found under {data_dir}")

    parts = [pd.read_csv(path) for path in files]
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.rename(
        columns={
            "play_duration": "play_duration_stat",
            "follow_user_num": "follow_user_num_stat",
        }
    )
    frame = frame.drop_duplicates(subset=["video_id"], keep="first")
    logger.info("KuaiRand video statistic读取完成。files=%s rows=%s", len(files), len(frame))
    return frame


def _attach_raw_features(
    standard: pd.DataFrame,
    random: pd.DataFrame,
    data_dir: Path,
    version_suffix: str,
    config: dict[str, Any],
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logger.info("开始合并KuaiRand原始特征...")

    user_features = pd.read_csv(data_dir / f"user_features_{version_suffix}.csv")
    video_basic = pd.read_csv(data_dir / f"video_features_basic_{version_suffix}.csv")
    video_basic = video_basic.rename(columns={"video_duration": "video_duration_basic"})
    include_aggregate_statistics = bool(config.get("raw_features", {}).get("include_aggregate_statistics", False))
    video_statistics = _load_video_statistics(data_dir, version_suffix, logger) if include_aggregate_statistics else None

    frames = [standard, random]
    for frame in frames:
        frame = frame.merge(user_features, on="user_id", how="left")
        frame = frame.merge(video_basic, on="video_id", how="left")
        if video_statistics is not None:
            frame = frame.merge(video_statistics, on="video_id", how="left")
        if frame is standard:
            standard = frame
        else:
            random = frame
    frames = [standard, random]

    sparse_names = [
        *[f"log::{column}" for column in KUAI_RAND_LOG_SPARSE_COLUMNS],
        *[f"user::{column}" for column in KUAI_RAND_USER_SPARSE_COLUMNS],
        *[f"video_basic::{column}" for column in KUAI_RAND_VIDEO_BASIC_SPARSE_COLUMNS],
    ]
    sparse_columns = [
        *KUAI_RAND_LOG_SPARSE_COLUMNS,
        *KUAI_RAND_USER_SPARSE_COLUMNS,
        *KUAI_RAND_VIDEO_BASIC_SPARSE_COLUMNS,
    ]
    raw_sparse_columns, raw_sparse_cardinalities, sparse_audit = _encode_sparse_columns(frames, sparse_columns, logger)

    statistic_numeric_columns = KUAI_RAND_VIDEO_STATISTIC_NUMERIC_COLUMNS if include_aggregate_statistics else []
    numeric_names = [
        *[f"log::{column}" for column in KUAI_RAND_LOG_NUMERIC_COLUMNS],
        *[f"user::{column}" for column in KUAI_RAND_USER_NUMERIC_COLUMNS],
        *[f"video_basic::{column}" for column in KUAI_RAND_VIDEO_BASIC_NUMERIC_COLUMNS],
        *[f"video_statistic::{column}" for column in statistic_numeric_columns],
    ]
    numeric_columns = [
        *KUAI_RAND_LOG_NUMERIC_COLUMNS,
        *KUAI_RAND_USER_NUMERIC_COLUMNS,
        *KUAI_RAND_VIDEO_BASIC_NUMERIC_COLUMNS,
        *statistic_numeric_columns,
    ]
    raw_numeric_columns, numeric_audit = _encode_numeric_columns(frames, numeric_columns, logger)

    sequence_length = int(config.get("raw_features", {}).get("sequence_length", 8))
    sequence_parts = [frame["tag"].map(_parse_feature_sequence) if "tag" in frame else pd.Series([[]] * len(frame)) for frame in frames]
    token_values = sorted({token for sequences in sequence_parts for values in sequences for token in values})
    token_mapping = {token: idx + 1 for idx, token in enumerate(token_values)}
    for frame, sequences in zip(frames, sequence_parts):
        for pos in range(sequence_length):
            value_column = f"{RAW_SEQUENCE_PREFIX}{pos}"
            mask_column = f"{RAW_SEQUENCE_MASK_PREFIX}{pos}"
            frame[value_column] = sequences.map(lambda values, pos=pos: token_mapping.get(values[pos], 0) if pos < len(values) else 0).astype(np.int64)
            frame[mask_column] = (frame[value_column].to_numpy(copy=False) > 0).astype(np.float32)

    sequence_non_empty = int(sum(sequences.map(bool).sum() for sequences in sequence_parts))
    meta = {
        "enabled": True,
        "raw_sparse_columns": raw_sparse_columns,
        "raw_sparse_feature_names": sparse_names,
        "raw_sparse_cardinalities": [int(value) for value in raw_sparse_cardinalities],
        "num_raw_sparse_features": int(len(raw_sparse_columns)),
        "raw_numeric_columns": raw_numeric_columns,
        "raw_numeric_feature_names": numeric_names,
        "num_raw_numeric_features": int(len(raw_numeric_columns)),
        "raw_sequence_columns": [f"{RAW_SEQUENCE_PREFIX}{pos}" for pos in range(sequence_length)],
        "raw_sequence_mask_columns": [f"{RAW_SEQUENCE_MASK_PREFIX}{pos}" for pos in range(sequence_length)],
        "raw_sequence_feature_name": "video_basic::tag",
        "raw_sequence_cardinality": int(len(token_mapping) + 1),
        "raw_sequence_length": int(sequence_length),
        "sparse_audit": sparse_audit,
        "numeric_audit": numeric_audit,
        "sequence_audit": {
            "non_empty_rows": sequence_non_empty,
            "token_cardinality": int(len(token_mapping) + 1),
        },
        "include_aggregate_statistics": include_aggregate_statistics,
        "excluded_same_exposure_feedback_columns": [
            "is_click",
            "is_like",
            "is_follow",
            "is_comment",
            "is_forward",
            "is_hate",
            "long_view",
            "profile_stay_time",
            "comment_stay_time",
            "is_profile_enter",
        ],
    }
    logger.info(
        "KuaiRand原始特征合并完成。sparse=%s numeric=%s seq_len=%s",
        meta["num_raw_sparse_features"],
        meta["num_raw_numeric_features"],
        meta["raw_sequence_length"],
    )
    return standard, random, meta


def _read_supplemental_csv(
    path: Path,
    usecols: list[str],
    dtype: dict[str, str],
    max_video_id: int,
    assume_sorted_contiguous: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Supplemental metadata file not found: {path}")
    if assume_sorted_contiguous:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, nrows=max_video_id + 1)

    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=1_000_000):
        parts.append(chunk.loc[chunk["final_video_id"] <= max_video_id].copy())
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def _normalize_category_ids(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    unknown_values = {"", "-124", "UNKNOWN", "nan", "NaN", "None", "<NA>"}
    return text.mask(text.isin(unknown_values), "")


def _encode_category_level(series: pd.Series) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    normalized = _normalize_category_ids(series)
    tokens = sorted(token for token in normalized.drop_duplicates().tolist() if token)
    mapping = {token: idx + 1 for idx, token in enumerate(tokens)}
    encoded = normalized.map(mapping).fillna(0).astype(np.int64).to_numpy()
    audit = {
        "unique_non_unknown_values": int(len(tokens)),
        "unknown_or_missing_rows": int((normalized == "").sum()),
    }
    return encoded, mapping, audit


def _build_item_content_features(
    config: dict[str, Any],
    item_lookup: pd.DataFrame,
    output_dir: Path,
    logger,
) -> dict[str, Any]:
    content_cfg = config.get("content_features", {})
    enabled = bool(content_cfg.get("enabled", False))
    num_items = int(len(item_lookup))
    if not enabled:
        np.savez(
            output_dir / "item_content.npz",
            content_category=np.zeros((num_items, 0), dtype=np.int64),
            content_numeric=np.zeros((num_items, 0), dtype=np.float32),
        )
        return {
            "enabled": False,
            "num_content_category_levels": 0,
            "content_category_cardinalities": [],
            "num_content_numeric_features": 0,
            "content_numeric_names": [],
            "item_content_path": str(output_dir / "item_content.npz"),
        }

    item_content_path = output_dir / "item_content.npz"
    if bool(content_cfg.get("reuse_existing_item_content", False)) and item_content_path.exists():
        logger.info("Reusing existing item content metadata from %s", item_content_path)

        existing = np.load(item_content_path, allow_pickle=False)
        content_category = existing["content_category"][:, : len(CATEGORY_ID_COLUMNS)]
        content_numeric = existing["content_numeric"][:, : len(CONTENT_NUMERIC_NAMES)]
        np.savez(item_content_path, content_category=content_category, content_numeric=content_numeric)

        return {
            "enabled": True,
            "categories_path": str(Path(content_cfg["categories_path"]).expanduser().resolve()),
            "assume_sorted_contiguous_final_video_id": bool(content_cfg.get("assume_sorted_contiguous_final_video_id", True)),
            "max_video_id_loaded": int(item_lookup["video_id"].max()),
            "num_content_category_levels": int(content_category.shape[1]),
            "content_category_names": [*CATEGORY_ID_COLUMNS],
            "content_category_cardinalities": [
                int(content_category[:, idx].max()) + 1 for idx in range(content_category.shape[1])
            ],
            "num_content_numeric_features": int(content_numeric.shape[1]),
            "content_numeric_names": CONTENT_NUMERIC_NAMES,
            "item_content_path": str(item_content_path),
            "reused_existing_item_content": True,
        }

    logger.info("Loading supplemental content metadata")
    max_video_id = int(item_lookup["video_id"].max())
    assume_sorted = bool(content_cfg.get("assume_sorted_contiguous_final_video_id", True))

    categories_path = Path(content_cfg["categories_path"]).expanduser().resolve()
    categories = _read_supplemental_csv(
        categories_path,
        ["final_video_id", *CATEGORY_ID_COLUMNS, *CATEGORY_PROB_COLUMNS],
        {column: "string" for column in CATEGORY_ID_COLUMNS},
        max_video_id,
        assume_sorted,
    )

    metadata = item_lookup.merge(categories, left_on="video_id", right_on="final_video_id", how="left")
    metadata.sort_values("item_idx", inplace=True, kind="mergesort")

    category_arrays: list[np.ndarray] = []
    category_cardinalities: list[int] = []
    category_audit: dict[str, dict[str, int]] = {}
    for column in CATEGORY_ID_COLUMNS:
        encoded, mapping, audit = _encode_category_level(metadata[column])
        category_arrays.append(encoded)
        category_cardinalities.append(len(mapping) + 1)
        category_audit[column] = audit

    prob_numeric = metadata[CATEGORY_PROB_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad_prob_rows = int(((prob_numeric < 0) | (prob_numeric > 1)).any(axis=1).sum())
    prob_numeric = prob_numeric.fillna(0.0).clip(0.0, 1.0).to_numpy(np.float32)

    content_numeric = prob_numeric.astype(np.float32, copy=False)
    content_category = np.column_stack(category_arrays).astype(np.int64)

    np.savez(
        item_content_path,
        content_category=content_category,
        content_numeric=content_numeric,
    )

    return {
        "enabled": True,
        "categories_path": str(categories_path),
        "assume_sorted_contiguous_final_video_id": assume_sorted,
        "max_video_id_loaded": max_video_id,
        "num_content_category_levels": int(content_category.shape[1]),
        "content_category_names": [*CATEGORY_ID_COLUMNS],
        "content_category_cardinalities": [int(value) for value in category_cardinalities],
        "num_content_numeric_features": int(content_numeric.shape[1]),
        "content_numeric_names": CONTENT_NUMERIC_NAMES,
        "item_content_path": str(item_content_path),
        "category_audit": category_audit,
        "bad_category_probability_rows_after_join": bad_prob_rows,
    }


def _matrix_from_columns(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    matrix = df[cols].to_numpy(np.float32)
    for idx, col in enumerate(cols):
        if col in {"session_pos", "elapsed_session_sec", "prev_gap_sec", "prev_play_sec"}:
            matrix[:, idx] = np.log1p(np.clip(matrix[:, idx], 0, None))
    return matrix


def _numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    return _matrix_from_columns(df, EXPOSURE_NUMERIC_NAMES)


def _session_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    return _matrix_from_columns(df, SESSION_NUMERIC_NAMES)


def _arrays_for_split(df: pd.DataFrame, raw_meta: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    arrays = {
        "user_idx": df["user_idx"].to_numpy(np.int64),
        "item_idx": df["item_idx"].to_numpy(np.int64),
        "tab": df["tab"].clip(lower=0, upper=14).to_numpy(np.int64),
        "prev_event": df["prev_event"].clip(lower=0, upper=UNKNOWN_PREV_EVENT).to_numpy(np.int64),
        "numeric": _numeric_matrix(df),
        "session_numeric": _session_numeric_matrix(df),
        "duration_sec": df["duration_sec"].to_numpy(np.float32),
        "event": df["event"].to_numpy(np.int64),
        "watch_event": df["watch_event"].to_numpy(np.int64),
        "time_bin": df["time_bin"].to_numpy(np.int64),
        "engagement_mark": df["engagement_mark"].to_numpy(np.float32),
        "play_time_sec": df["play_time_sec"].to_numpy(np.float32),
        "session_key": df["session_key"].to_numpy(np.int64),
    }
    if raw_meta and raw_meta.get("enabled", False):
        raw_sparse_columns = list(raw_meta.get("raw_sparse_columns", []))
        raw_numeric_columns = list(raw_meta.get("raw_numeric_columns", []))
        raw_sequence_columns = list(raw_meta.get("raw_sequence_columns", []))
        raw_sequence_mask_columns = list(raw_meta.get("raw_sequence_mask_columns", []))
        arrays["raw_sparse"] = (
            df[raw_sparse_columns].to_numpy(np.int64, copy=False)
            if raw_sparse_columns
            else np.zeros((len(df), 0), dtype=np.int64)
        )
        arrays["raw_numeric"] = (
            df[raw_numeric_columns].to_numpy(np.float32, copy=False)
            if raw_numeric_columns
            else np.zeros((len(df), 0), dtype=np.float32)
        )
        arrays["raw_seq"] = (
            df[raw_sequence_columns].to_numpy(np.int64, copy=False)
            if raw_sequence_columns
            else np.zeros((len(df), 0), dtype=np.int64)
        )
        arrays["raw_seq_mask"] = (
            df[raw_sequence_mask_columns].to_numpy(np.float32, copy=False)
            if raw_sequence_mask_columns
            else np.zeros((len(df), 0), dtype=np.float32)
        )
    else:
        arrays["raw_sparse"] = np.zeros((len(df), 0), dtype=np.int64)
        arrays["raw_numeric"] = np.zeros((len(df), 0), dtype=np.float32)
        arrays["raw_seq"] = np.zeros((len(df), 0), dtype=np.int64)
        arrays["raw_seq_mask"] = np.zeros((len(df), 0), dtype=np.float32)
    return arrays


def _column_values_for_mask(frame: pd.DataFrame, column: str, mask: np.ndarray, dtype: Any) -> np.ndarray:
    values = frame[column].to_numpy(copy=False)
    return values[mask].astype(dtype, copy=False)


def _matrix_from_frame_mask(frame: pd.DataFrame, cols: list[str], mask: np.ndarray) -> np.ndarray:

    columns: list[np.ndarray] = []
    for col in cols:
        value = frame[col].to_numpy(copy=False)[mask].astype(np.float32, copy=False)
        if col in {"session_pos", "elapsed_session_sec", "prev_gap_sec", "prev_play_sec"}:
            value = np.log1p(np.clip(value, 0, None)).astype(np.float32, copy=False)
        columns.append(value)

    return np.column_stack(columns).astype(np.float32, copy=False)


def _arrays_for_split_mask(
    frame: pd.DataFrame,
    mask: np.ndarray,
    unknown_prev_event: int,
    raw_meta: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:

    arrays = {
        "user_idx": _column_values_for_mask(frame, "user_idx", mask, np.int64),
        "item_idx": _column_values_for_mask(frame, "item_idx", mask, np.int64),
        "tab": np.clip(_column_values_for_mask(frame, "tab", mask, np.int64), 0, 14),
        "prev_event": np.clip(_column_values_for_mask(frame, "prev_event", mask, np.int64), 0, int(unknown_prev_event)),
        "numeric": _matrix_from_frame_mask(frame, EXPOSURE_NUMERIC_NAMES, mask),
        "session_numeric": _matrix_from_frame_mask(frame, SESSION_NUMERIC_NAMES, mask),
        "duration_sec": _column_values_for_mask(frame, "duration_sec", mask, np.float32),
        "event": _column_values_for_mask(frame, "event", mask, np.int64),
        "watch_event": _column_values_for_mask(frame, "watch_event", mask, np.int64),
        "time_bin": _column_values_for_mask(frame, "time_bin", mask, np.int64),
        "engagement_mark": _column_values_for_mask(frame, "engagement_mark", mask, np.float32),
        "play_time_sec": _column_values_for_mask(frame, "play_time_sec", mask, np.float32),
        "session_key": _column_values_for_mask(frame, "session_key", mask, np.int64),
    }

    if raw_meta and raw_meta.get("enabled", False):
        raw_sparse_columns = list(raw_meta.get("raw_sparse_columns", []))
        raw_numeric_columns = list(raw_meta.get("raw_numeric_columns", []))
        raw_sequence_columns = list(raw_meta.get("raw_sequence_columns", []))
        raw_sequence_mask_columns = list(raw_meta.get("raw_sequence_mask_columns", []))
        row_count = int(mask.sum())
        arrays["raw_sparse"] = (
            frame[raw_sparse_columns].to_numpy(np.int64, copy=False)[mask]
            if raw_sparse_columns
            else np.zeros((row_count, 0), dtype=np.int64)
        )
        arrays["raw_numeric"] = (
            frame[raw_numeric_columns].to_numpy(np.float32, copy=False)[mask]
            if raw_numeric_columns
            else np.zeros((row_count, 0), dtype=np.float32)
        )
        arrays["raw_seq"] = (
            frame[raw_sequence_columns].to_numpy(np.int64, copy=False)[mask]
            if raw_sequence_columns
            else np.zeros((row_count, 0), dtype=np.int64)
        )
        arrays["raw_seq_mask"] = (
            frame[raw_sequence_mask_columns].to_numpy(np.float32, copy=False)[mask]
            if raw_sequence_mask_columns
            else np.zeros((row_count, 0), dtype=np.float32)
        )
    else:
        row_count = int(mask.sum())
        arrays["raw_sparse"] = np.zeros((row_count, 0), dtype=np.int64)
        arrays["raw_numeric"] = np.zeros((row_count, 0), dtype=np.float32)
        arrays["raw_seq"] = np.zeros((row_count, 0), dtype=np.int64)
        arrays["raw_seq_mask"] = np.zeros((row_count, 0), dtype=np.float32)

    return arrays


def _save_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def prepare_kuairand(config: dict[str, Any], logger) -> dict[str, Any]:
    data_dir = Path(config["data"]["raw_dir"]).expanduser().resolve() / "data"
    output_dir = Path(config["data"]["processed_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading KuaiRand logs from %s", data_dir)
    standard, random = _load_logs(data_dir, config["data"]["version_suffix"], config["debug"]["max_rows"])
    event_names = _event_names(config)
    watch_event_names = _watch_event_names(config)
    unknown_prev_event = _unknown_prev_event(config)

    upper_bounds = build_time_bin_upper_bounds(
        config["time_bins"]["linear_seconds"],
        config["time_bins"]["max_seconds"],
        config["time_bins"]["exponential_growth"],
    )

    logger.info("Constructing standard event taxonomy")
    standard, standard_audit = _construct_events(standard, config, upper_bounds)
    logger.info("Constructing random event taxonomy")
    random, random_audit = _construct_events(random, config, upper_bounds)

    logger.info("开始处理KuaiRand原始特征开关...")
    if bool(config.get("raw_features", {}).get("enabled", True)):

        logger.info("Attaching raw KuaiRand features")
        standard, random, raw_feature_meta = _attach_raw_features(
            standard,
            random,
            data_dir,
            config["data"]["version_suffix"],
            config,
            logger,
        )
    else:

        raw_feature_meta = {
            "enabled": False,
            "raw_sparse_columns": [],
            "raw_sparse_feature_names": [],
            "raw_sparse_cardinalities": [],
            "num_raw_sparse_features": 0,
            "raw_numeric_columns": [],
            "raw_numeric_feature_names": [],
            "num_raw_numeric_features": 0,
            "raw_sequence_columns": [],
            "raw_sequence_mask_columns": [],
            "raw_sequence_feature_name": "disabled",
            "raw_sequence_cardinality": 0,
            "raw_sequence_length": 0,
            "include_aggregate_statistics": False,
        }
        logger.info("KuaiRand原始特征已跳过，使用clean schema。")
    logger.info("KuaiRand原始特征开关处理完成。enabled=%s", bool(raw_feature_meta.get("enabled", False)))

    logger.info("Encoding user and item ids")
    cardinality, item_lookup = _encode_ids(standard, random)

    standard["split"] = _assign_configured_splits(standard, config)
    random["split"] = _assign_configured_splits(random, config)

    content_meta = _build_item_content_features(config, item_lookup, output_dir, logger)

    logger.info("Writing processed npz files to %s", output_dir)
    compressed = bool(config["data"]["compressed_npz"])
    split_counts: dict[str, int] = {}
    for source_name, frame in [("standard", standard), ("random", random)]:
        for split_name in ["train", "val", "test"]:
            logger.info("Writing %s_%s split", source_name, split_name)

            mask = (frame["split"].to_numpy(copy=False) == split_name)
            split_counts[f"{source_name}_{split_name}"] = int(mask.sum())

            arrays = _arrays_for_split_mask(frame, mask, unknown_prev_event, raw_feature_meta)
            _save_npz(output_dir / f"{source_name}_{split_name}.npz", arrays, compressed)
            del arrays

    meta = {
        "dataset": config["data"]["name"],
        "raw_dir": str(Path(config["data"]["raw_dir"]).expanduser().resolve()),
        "processed_dir": str(output_dir),
        "contract_path": str(Path(config["contract_path"]).expanduser().resolve()),
        "event_names": event_names,
        "watch_event_names": watch_event_names,
        "unknown_prev_event": unknown_prev_event,
        "time_bin_upper_bounds_sec": upper_bounds.tolist(),
        "num_time_bins": int(len(upper_bounds)),
        **cardinality,
        "num_tabs": 15,
        "num_numeric_features": len(EXPOSURE_NUMERIC_NAMES),
        "numeric_feature_names": EXPOSURE_NUMERIC_NAMES,
        "num_session_numeric_features": len(SESSION_NUMERIC_NAMES),
        "session_numeric_feature_names": SESSION_NUMERIC_NAMES,
        "content_features": content_meta,
        "raw_features": raw_feature_meta,
        "standard_audit": standard_audit,
        "random_audit": random_audit,
        "split_counts": split_counts,
        "split_policy": config["split"],
        "event_taxonomy": config["event_taxonomy"],
        "preprocessing": config["preprocessing"],
        "contract_deviation_flags": {
            "requires_contract_v2_for_invalid_duration_policy": bool(
                config["preprocessing"]["invalid_duration_policy"] != "fail"
                and (standard_audit["invalid_duration_rows"] > 0 or random_audit["invalid_duration_rows"] > 0)
            ),
            "reason": "The V1 contract does not define event taxonomy behavior for duration_ms <= 0.",
        },
    }

    meta = portableize_serialized_paths(meta)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    logger.info("Preparation complete: %s", split_counts)
    return meta
