from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsi.utils.bridge import portableize_serialized_paths
from dsi.utils.time_bins import assign_time_bins, build_time_bin_upper_bounds

WATCH_EVENT_NAMES = ["early_skip", "mid_watch", "true_completion", "overplay_replay"]
EVENT_NAMES = list(WATCH_EVENT_NAMES)
UNKNOWN_PREV_EVENT = len(EVENT_NAMES)
ACTION_COLUMNS = [
    "userid",
    "feedid",
    "date_",
    "device",
    "read_comment",
    "comment",
    "like",
    "play",
    "stay",
    "click_avatar",
    "forward",
    "follow",
    "favorite",
]
FEED_COLUMNS = [
    "feedid",
    "authorid",
    "videoplayseconds",
    "bgm_song_id",
    "bgm_singer_id",
    "manual_tag_list",
    "machine_tag_list",
]
NUMERIC_NAMES = [
    "log_duration",
    "session_pos",
    "elapsed_session_sec",
    "prev_gap_sec",
    "prev_play_sec",
    "hour_sin",
    "hour_cos",
    "user_seen_log",
    "user_play_mean_prev_log",
    "user_ratio_mean_prev",
    "user_completion_rate_prev",
    "user_overplay_rate_prev",
    "user_engagement_rate_prev",
    "user_like_rate_prev",
    "user_read_comment_rate_prev",
    "user_favorite_rate_prev",
    "user_forward_rate_prev",
]
SESSION_NUMERIC_NAMES = [
    "session_pos",
    "elapsed_session_sec",
    "prev_gap_sec",
    "prev_play_sec",
    "hour_sin",
    "hour_cos",
]
CONTENT_BASE_CATEGORY_NAMES = ["authorid", "bgm_song_id", "bgm_singer_id"]
HISTORY_NUMERIC_NAMES = [
    "user_seen_log",
    "user_play_mean_prev_log",
    "user_ratio_mean_prev",
    "user_completion_rate_prev",
    "user_overplay_rate_prev",
    "user_engagement_rate_prev",
    "user_like_rate_prev",
    "user_read_comment_rate_prev",
    "user_favorite_rate_prev",
    "user_forward_rate_prev",
]
RAW_FEATURE_META_DISABLED = {
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


def _read_user_action(config: dict[str, Any], logger) -> pd.DataFrame:
    logger.info("Reading WeChat21 exposure logs.")

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"]).expanduser().resolve()
    path = raw_dir / str(data_cfg.get("user_action_file", "user_action.csv"))
    nrows = config.get("debug", {}).get("max_rows", {}).get("standard_per_file")
    if not path.exists():
        raise FileNotFoundError(f"WeChat21 user_action not found: {path}")

    frame = pd.read_csv(path, usecols=ACTION_COLUMNS, nrows=nrows)
    frame.rename(
        columns={
            "userid": "user_id",
            "feedid": "video_id",
            "date_": "date",
            "play": "play_time_ms",
        },
        inplace=True,
    )

    logger.info("WeChat21 exposure logs loaded. rows=%s", int(len(frame)))
    return frame


def _read_feed_info(config: dict[str, Any], logger) -> pd.DataFrame:
    logger.info("Reading WeChat21 feed metadata.")

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"]).expanduser().resolve()
    path = raw_dir / str(data_cfg.get("feed_info_file", "feed_info.csv"))
    if not path.exists():
        raise FileNotFoundError(f"WeChat21 feed_info not found: {path}")

    feed = pd.read_csv(path, usecols=FEED_COLUMNS)
    feed.rename(columns={"feedid": "video_id", "videoplayseconds": "duration_sec"}, inplace=True)
    feed = feed.drop_duplicates("video_id", keep="first").copy()

    logger.info("WeChat21 feed metadata loaded. rows=%s", int(len(feed)))
    return feed


def _numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _sanitize_and_join(frame: pd.DataFrame, feed: pd.DataFrame, config: dict[str, Any], logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    logger.info("Merging WeChat21 duration fields.")

    before_rows = int(len(frame))
    frame["user_id"] = _numeric(frame["user_id"]).astype(np.int64)
    frame["video_id"] = _numeric(frame["video_id"]).astype(np.int64)
    frame["date"] = _numeric(frame["date"]).astype(np.int64)
    frame["device"] = _numeric(frame["device"]).astype(np.int64)
    prep = config.get("preprocessing", {})
    label_column = str(prep.get("label_column", "play"))
    if label_column == "play":
        label_source_column = "play_time_ms"
    elif label_column == "stay":
        label_source_column = "stay"
    else:
        raise ValueError(f"Unsupported WeChat21 label_column: {label_column}")
    frame["play_time_ms"] = _numeric(frame[label_source_column]).clip(lower=0).astype(np.float32)
    feed["video_id"] = _numeric(feed["video_id"]).astype(np.int64)
    feed["duration_sec"] = _numeric(feed["duration_sec"]).astype(np.float32)
    frame = frame.merge(feed[["video_id", "duration_sec"]], on="video_id", how="left")

    min_duration_ms = int(prep.get("min_duration_ms", 1))
    duration_ms = (frame["duration_sec"] * 1000.0).astype(np.float32)
    invalid_duration = duration_ms.isna() | (duration_ms <= 0)
    policy = str(prep.get("invalid_duration_policy", "drop"))
    if policy == "drop":
        frame = frame.loc[~invalid_duration].copy()
        duration_ms = duration_ms.loc[~invalid_duration]
    elif invalid_duration.any():
        raise ValueError(f"Found invalid WeChat21 duration rows: {int(invalid_duration.sum())}")
    frame["duration_ms_safe"] = duration_ms.clip(lower=min_duration_ms).astype(np.float32)

    after_duration_rows = int(len(frame))
    non_positive_label_rows = 0
    if bool(prep.get("require_positive_label", False)):
        positive_label = frame["play_time_ms"].to_numpy(np.float64) > 0.0
        non_positive_label_rows = int((~positive_label).sum())
        frame = frame.loc[positive_label].copy()

    duration_cap_rows = 0
    max_duration_sec_exclusive = prep.get("max_duration_sec_exclusive")
    if max_duration_sec_exclusive is not None:
        logger.info("Filtering WeChat21 samples above the duration limit. max_duration_sec_exclusive=%s", max_duration_sec_exclusive)
        keep = (frame["duration_ms_safe"].to_numpy(np.float64) / 1000.0) < float(max_duration_sec_exclusive)
        duration_cap_rows = int((~keep).sum())
        frame = frame.loc[keep].copy()
        logger.info("Filtered WeChat21 samples above the duration limit. removed=%s", duration_cap_rows)

    max_play_ratio = prep.get("max_play_ratio")
    over_ratio_rows = 0
    if max_play_ratio is not None:
        ratio = frame["play_time_ms"].to_numpy(np.float64) / np.maximum(frame["duration_ms_safe"].to_numpy(np.float64), 1.0)
        keep = ratio < float(max_play_ratio)
        over_ratio_rows = int((~keep).sum())
        frame = frame.loc[keep].copy()

    audit = {
        "label_column": label_column,
        "rows_before_join": before_rows,
        "missing_or_invalid_duration_rows": int(invalid_duration.sum()),
        "rows_after_duration_policy": after_duration_rows,
        "non_positive_label_rows_removed": non_positive_label_rows,
        "duration_cap_rows_removed": duration_cap_rows,
        "max_duration_sec_exclusive": max_duration_sec_exclusive,
        "over_max_play_ratio_rows_removed": over_ratio_rows,
        "rows_after_filters": int(len(frame)),
    }
    logger.info("WeChat21 duration fields merged. audit=%s", audit)
    return frame.reset_index(drop=True), audit


def _construct_events_and_sessions(frame: pd.DataFrame, config: dict[str, Any], upper_bounds: np.ndarray, logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    logger.info("Building WeChat21 event labels and session fields.")

    taxonomy = config["event_taxonomy"]
    alpha = float(taxonomy.get("alpha", 0.3))
    mid_cut = float(taxonomy.get("mid_completion_cut", 0.9))
    beta = float(taxonomy.get("beta", 1.1))
    ratio = frame["play_time_ms"].to_numpy(np.float64) / np.maximum(frame["duration_ms_safe"].to_numpy(np.float64), 1.0)
    watch_event = np.zeros(len(frame), dtype=np.int64)
    watch_event[(ratio >= alpha) & (ratio < mid_cut)] = 1
    watch_event[(ratio >= mid_cut) & (ratio <= beta)] = 2
    watch_event[ratio > beta] = 3
    frame["watch_event"] = watch_event
    frame["event"] = watch_event

    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame.sort_values(["user_id", "date", "_row_order"], inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
    day_start = (frame["date"].to_numpy(np.int64) - int(frame["date"].min())) * 86400.0
    in_day_pos = frame.groupby(["user_id", "date"], sort=False).cumcount().to_numpy(np.float32)
    frame["timestamp"] = (day_start + in_day_pos).astype(np.float32)

    grouped = frame.groupby("user_id", sort=False)
    prev_ts = grouped["timestamp"].shift(1)
    next_ts = grouped["timestamp"].shift(-1)
    gap_sec = (frame["timestamp"] - prev_ts).fillna(0).clip(lower=0)
    next_gap_sec = next_ts - frame["timestamp"]
    gamma_sec = float(taxonomy.get("gamma_minutes", 30)) * 60.0
    new_session = prev_ts.isna() | (gap_sec > gamma_sec)
    frame["session_ordinal"] = new_session.groupby(frame["user_id"], sort=False).cumsum().astype(np.int64) - 1
    session_index = pd.MultiIndex.from_arrays([frame["user_id"].to_numpy(), frame["session_ordinal"].to_numpy()])
    frame["session_key"] = pd.factorize(session_index, sort=False)[0].astype(np.int64)
    session_group = frame.groupby("session_key", sort=False)
    frame["session_pos"] = session_group.cumcount().astype(np.int32)
    frame["elapsed_session_sec"] = (frame["timestamp"] - session_group["timestamp"].transform("first")).clip(lower=0).astype(np.float32)
    frame["prev_gap_sec"] = np.where(new_session.to_numpy(), 0.0, gap_sec.clip(upper=gamma_sec)).astype(np.float32)
    frame["prev_play_sec"] = (session_group["play_time_ms"].shift(1).fillna(0).clip(lower=0) / 1000.0).astype(np.float32)
    frame["prev_event"] = session_group["event"].shift(1).fillna(UNKNOWN_PREV_EVENT).astype(np.int64)
    frame["is_session_end"] = (next_ts.isna() | (next_gap_sec > gamma_sec)).astype(np.float32)

    frame["time_bin"] = assign_time_bins(frame["play_time_ms"].to_numpy(np.float64) / 1000.0, upper_bounds)
    frame["duration_sec"] = (frame["duration_ms_safe"] / 1000.0).astype(np.float32)
    frame["play_time_sec"] = (frame["play_time_ms"].clip(lower=0) / 1000.0).astype(np.float32)
    frame["log_duration"] = np.log1p(frame["duration_sec"]).astype(np.float32)
    frame["hour_sin"] = 0.0
    frame["hour_cos"] = 1.0
    frame["tab"] = frame["device"].clip(lower=0).astype(np.int64)
    feedback_columns = ["like", "comment", "forward", "follow", "favorite", "read_comment", "click_avatar"]
    for column in feedback_columns:
        frame[column] = _numeric(frame[column]).clip(lower=0).astype(np.float32)
    frame["engagement_mark"] = (frame[feedback_columns].max(axis=1) > 0).astype(np.float32)
    _add_user_history_features(frame, config, logger)

    watch_counts = pd.Series(frame["watch_event"]).value_counts().reindex(range(len(WATCH_EVENT_NAMES)), fill_value=0)
    audit = {
        "sessions": int(frame["session_key"].nunique()),
        "watch_event_counts": {WATCH_EVENT_NAMES[idx]: int(value) for idx, value in watch_counts.items()},
        "session_end_count": int(frame["is_session_end"].sum()),
        "timestamp_note": "WeChat21 has no intra-day exposure timestamp; timestamp is reconstructed from user-date row order.",
    }
    logger.info("WeChat21 event labels and session fields built. sessions=%s", audit["sessions"])
    return frame, audit


def _encode_ids(frame: pd.DataFrame, logger) -> dict[str, int]:
    logger.info("Encoding WeChat21 user and video IDs.")

    frame["user_idx"] = pd.factorize(frame["user_id"], sort=True)[0].astype(np.int64)
    frame["item_idx"] = pd.factorize(frame["video_id"], sort=True)[0].astype(np.int64)

    cardinality = {
        "num_users": int(frame["user_idx"].max()) + 1,
        "num_items": int(frame["item_idx"].max()) + 1,
    }
    logger.info("WeChat21 user and video IDs encoded. users=%s items=%s", cardinality["num_users"], cardinality["num_items"])
    return cardinality


def _assign_row_shuffle_splits(frame: pd.DataFrame, ratios: list[float], random_state: int, logger) -> pd.Series:
    logger.info("Creating the standard WeChat21 split.")

    shuffled_index = frame.sample(frac=1.0, random_state=int(random_state)).index.to_numpy()
    n_rows = int(len(shuffled_index))
    n_train = int(float(ratios[0]) * n_rows)
    n_val_end = int(float(ratios[0] + ratios[1]) * n_rows)

    split = pd.Series("test", index=frame.index, dtype=object)
    split.loc[shuffled_index[:n_train]] = "train"
    split.loc[shuffled_index[n_train:n_val_end]] = "val"
    split.loc[shuffled_index[n_val_end:]] = "test"
    logger.info(
        "Standard WeChat21 split created. train=%s val=%s test=%s",
        int((split == "train").sum()),
        int((split == "val").sum()),
        int((split == "test").sum()),
    )
    return split


def _normalize_sparse_value(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    unknown_values = {"", "nan", "NaN", "None", "<NA>", "UNKNOWN", "unknown"}
    return text.mask(text.isin(unknown_values), "")


def _encode_category_level(series: pd.Series) -> tuple[np.ndarray, int, dict[str, int]]:

    normalized = _normalize_sparse_value(series)
    tokens = sorted(token for token in normalized.drop_duplicates().tolist() if token)
    mapping = {token: idx + 1 for idx, token in enumerate(tokens)}
    encoded = normalized.map(mapping).fillna(0).astype(np.int64).to_numpy()

    audit = {
        "unique_non_unknown_values": int(len(tokens)),
        "unknown_or_missing_rows": int((normalized == "").sum()),
    }
    return encoded, int(len(mapping) + 1), audit


def _manual_tag(value: object, pos: int) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text in {"nan", "NaN", "None"}:
        return ""
    parts = [part.strip() for part in text.replace("|", ";").split(";") if part.strip()]
    return parts[int(pos)] if int(pos) < len(parts) else ""


def _machine_tag(value: object, pos: int) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text in {"nan", "NaN", "None"}:
        return ""
    parts = [part.strip() for part in text.split(";") if part.strip()]
    if int(pos) >= len(parts):
        return ""
    return parts[int(pos)].split()[0].strip()


def _add_user_history_features(frame: pd.DataFrame, config: dict[str, Any], logger) -> None:
    logger.info("Building WeChat21 user-history statistics.")

    history_cfg = config.get("user_history_features", {})
    if not bool(history_cfg.get("enabled", False)):
        for name in HISTORY_NUMERIC_NAMES:
            frame[name] = 0.0
        logger.info("WeChat21 user-history statistics disabled.")
        return

    grouped = frame.groupby("user_id", sort=False)
    count_prev = grouped.cumcount().to_numpy(np.float32)
    denom = np.maximum(count_prev, 1.0)
    frame["user_seen_log"] = np.log1p(count_prev).astype(np.float32)

    play = frame["play_time_sec"].to_numpy(np.float32)
    duration = np.maximum(frame["duration_sec"].to_numpy(np.float32), 1e-6)
    ratio = np.clip(play / duration, 0.0, 10.0).astype(np.float32)
    play_prev_sum = grouped["play_time_sec"].cumsum().to_numpy(np.float32) - play
    ratio_series = pd.Series(ratio, index=frame.index)
    ratio_prev_sum = ratio_series.groupby(frame["user_id"], sort=False).cumsum().to_numpy(np.float32) - ratio
    frame["user_play_mean_prev_log"] = np.log1p(play_prev_sum / denom).astype(np.float32)
    frame["user_ratio_mean_prev"] = (ratio_prev_sum / denom).astype(np.float32)

    completion = (frame["watch_event"].to_numpy(np.int64) == 2).astype(np.float32)
    overplay = (frame["watch_event"].to_numpy(np.int64) == 3).astype(np.float32)
    engagement = frame["engagement_mark"].to_numpy(np.float32)
    for name, values in [
        ("user_completion_rate_prev", completion),
        ("user_overplay_rate_prev", overplay),
        ("user_engagement_rate_prev", engagement),
        ("user_like_rate_prev", frame["like"].to_numpy(np.float32)),
        ("user_read_comment_rate_prev", frame["read_comment"].to_numpy(np.float32)),
        ("user_favorite_rate_prev", frame["favorite"].to_numpy(np.float32)),
        ("user_forward_rate_prev", frame["forward"].to_numpy(np.float32)),
    ]:
        series = pd.Series(values, index=frame.index)
        prev_sum = series.groupby(frame["user_id"], sort=False).cumsum().to_numpy(np.float32) - values
        frame[name] = (prev_sum / denom).astype(np.float32)

    logger.info("WeChat21 user-history statistics built. features=%s", len(HISTORY_NUMERIC_NAMES))


def _build_item_content_features(
    frame: pd.DataFrame,
    feed: pd.DataFrame,
    cardinality: dict[str, int],
    config: dict[str, Any],
    output_dir: Path,
    logger,
) -> dict[str, Any]:
    logger.info("Building WeChat21 content features.")

    num_items = int(cardinality["num_items"])
    item_content_path = output_dir / "item_content.npz"
    content_cfg = config.get("content_features", {})
    if not bool(content_cfg.get("enabled", True)):
        np.savez(
            item_content_path,
            content_category=np.zeros((num_items, 0), dtype=np.int64),
            content_numeric=np.zeros((num_items, 0), dtype=np.float32),
        )
        logger.info("WeChat21 content features disabled; writing empty item_content.")
        return {
            "enabled": False,
            "num_content_category_levels": 0,
            "content_category_cardinalities": [],
            "num_content_numeric_features": 0,
            "content_numeric_names": [],
            "item_content_path": str(item_content_path),
        }

    item_lookup = (
        frame[["video_id", "item_idx"]]
        .drop_duplicates("item_idx")
        .sort_values("item_idx", kind="mergesort")
    )
    aligned = item_lookup.merge(feed, on="video_id", how="left").sort_values("item_idx", kind="mergesort")
    if len(aligned) != num_items:
        raise ValueError(f"WeChat21 content alignment failed: rows={len(aligned)} items={num_items}")

    tag_slots = int(content_cfg.get("tag_slots", 4))
    content_category_names = [*CONTENT_BASE_CATEGORY_NAMES]
    for pos in range(tag_slots):
        col = f"manual_tag_{pos}"
        aligned[col] = aligned["manual_tag_list"].map(lambda value, pos=pos: _manual_tag(value, pos))
        content_category_names.append(col)
    for pos in range(tag_slots):
        col = f"machine_tag_{pos}"
        aligned[col] = aligned["machine_tag_list"].map(lambda value, pos=pos: _machine_tag(value, pos))
        content_category_names.append(col)

    category_arrays: list[np.ndarray] = []
    category_cardinalities: list[int] = []
    category_audit: dict[str, dict[str, int]] = {}
    for column in content_category_names:
        encoded, cardinality_value, audit = _encode_category_level(aligned[column])
        category_arrays.append(encoded)
        category_cardinalities.append(cardinality_value)
        category_audit[column] = audit

    content_category = np.column_stack(category_arrays).astype(np.int64, copy=False)
    content_numeric = np.zeros((num_items, 0), dtype=np.float32)
    content_numeric_names: list[str] = []
    np.savez(item_content_path, content_category=content_category, content_numeric=content_numeric)

    meta = {
        "enabled": True,
        "num_content_category_levels": int(content_category.shape[1]),
        "content_category_names": content_category_names,
        "content_category_cardinalities": [int(value) for value in category_cardinalities],
        "num_content_numeric_features": int(content_numeric.shape[1]),
        "content_numeric_names": content_numeric_names,
        "item_content_path": str(item_content_path),
        "category_audit": category_audit,
        "text_features_excluded": True,
    }
    logger.info("WeChat21 content features built. levels=%s", meta["num_content_category_levels"])
    return meta


def _matrix(frame: pd.DataFrame, cols: list[str], mask: np.ndarray) -> np.ndarray:

    columns: list[np.ndarray] = []
    for col in cols:
        value = frame[col].to_numpy(copy=False)[mask].astype(np.float32, copy=False)
        if col in {"session_pos", "elapsed_session_sec", "prev_gap_sec", "prev_play_sec"}:
            value = np.log1p(np.clip(value, 0, None)).astype(np.float32, copy=False)
        columns.append(value)

    return np.column_stack(columns).astype(np.float32, copy=False)


def _arrays_for_mask(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, np.ndarray]:

    row_count = int(mask.sum())
    arrays = {
        "user_idx": frame["user_idx"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "item_idx": frame["item_idx"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "tab": frame["tab"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "prev_event": np.clip(frame["prev_event"].to_numpy(copy=False)[mask], 0, UNKNOWN_PREV_EVENT).astype(np.int64, copy=False),
        "numeric": _matrix(frame, NUMERIC_NAMES, mask),
        "session_numeric": _matrix(frame, SESSION_NUMERIC_NAMES, mask),
        "duration_sec": frame["duration_sec"].to_numpy(copy=False)[mask].astype(np.float32, copy=False),
        "event": frame["event"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "watch_event": frame["watch_event"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "time_bin": frame["time_bin"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "engagement_mark": frame["engagement_mark"].to_numpy(copy=False)[mask].astype(np.float32, copy=False),
        "play_time_sec": frame["play_time_sec"].to_numpy(copy=False)[mask].astype(np.float32, copy=False),
        "session_key": frame["session_key"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
    }

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


def prepare_wechat21(config: dict[str, Any], logger) -> dict[str, Any]:
    logger.info("Preprocessing the WeChat21 benchmark.")

    output_dir = Path(config["data"]["processed_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    upper_bounds = build_time_bin_upper_bounds(
        config["time_bins"]["linear_seconds"],
        config["time_bins"]["max_seconds"],
        config["time_bins"]["exponential_growth"],
    )

    action = _read_user_action(config, logger)
    feed = _read_feed_info(config, logger)
    frame, sanitize_audit = _sanitize_and_join(action, feed, config, logger)
    del action
    frame, event_audit = _construct_events_and_sessions(frame, config, upper_bounds, logger)
    cardinality = _encode_ids(frame, logger)

    split_cfg = config.get("split", {})
    split_unit = str(split_cfg.get("unit", "row_shuffle_80_10_10"))
    if split_unit not in {"row_shuffle", "row_shuffle_80_10_10"}:
        raise ValueError(f"WeChat21 only supports row shuffle split, got {split_unit}")
    frame["split"] = _assign_row_shuffle_splits(
        frame,
        [float(value) for value in split_cfg.get("ratios", [0.8, 0.1, 0.1])],
        int(split_cfg.get("random_state", 1234)),
        logger,
    )
    content_meta = _build_item_content_features(frame, feed, cardinality, config, output_dir, logger)

    compressed = bool(config["data"].get("compressed_npz", False))
    split_counts: dict[str, int] = {}
    empty_mask = np.zeros(len(frame), dtype=bool)
    for split_name in ["train", "val", "test"]:
        mask = frame["split"].to_numpy(copy=False) == split_name
        split_counts[f"standard_{split_name}"] = int(mask.sum())
        arrays = _arrays_for_mask(frame, mask)
        _save_npz(output_dir / f"standard_{split_name}.npz", arrays, compressed)
        del arrays
        empty_arrays = _arrays_for_mask(frame, empty_mask)
        split_counts[f"random_{split_name}"] = 0
        _save_npz(output_dir / f"random_{split_name}.npz", empty_arrays, compressed)
        del empty_arrays

    manifest = {
        "dataset": config["data"]["name"],
        "raw_dir": str(Path(config["data"]["raw_dir"]).expanduser().resolve()),
        "processed_dir": str(output_dir),
        "contract_path": str(Path(config["contract_path"]).expanduser().resolve()),
        "event_names": EVENT_NAMES,
        "watch_event_names": WATCH_EVENT_NAMES,
        "unknown_prev_event": UNKNOWN_PREV_EVENT,
        "time_bin_upper_bounds_sec": upper_bounds.tolist(),
        "num_time_bins": int(len(upper_bounds)),
        **cardinality,
        "num_tabs": int(frame["tab"].max()) + 1,
        "num_numeric_features": len(NUMERIC_NAMES),
        "numeric_feature_names": NUMERIC_NAMES,
        "num_session_numeric_features": len(SESSION_NUMERIC_NAMES),
        "session_numeric_feature_names": SESSION_NUMERIC_NAMES,
        "content_features": content_meta,
        "raw_features": RAW_FEATURE_META_DISABLED,
        "standard_audit": {**sanitize_audit, **event_audit},
        "random_audit": {"rows_after_duration_policy": 0},
        "split_counts": split_counts,
        "split_policy": config["split"],
        "event_taxonomy": config["event_taxonomy"],
        "preprocessing": config["preprocessing"],
    }
    manifest = portableize_serialized_paths(manifest)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    logger.info("WeChat21 benchmark preprocessing complete. split_counts=%s", split_counts)
    return manifest
