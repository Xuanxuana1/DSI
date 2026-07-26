from __future__ import annotations

import ast
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsi.utils.bridge import portableize_serialized_paths
from dsi.utils.time_bins import assign_time_bins, build_time_bin_upper_bounds

WATCH_EVENT_NAMES = ["early_skip", "mid_watch", "true_completion", "overplay_replay"]
EVENT_NAMES = list(WATCH_EVENT_NAMES)
UNKNOWN_PREV_EVENT = len(EVENT_NAMES)
NUMERIC_NAMES = [
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
MATRIX_COLUMNS = [
    "user_id",
    "video_id",
    "play_duration",
    "video_duration",
    "time",
    "date",
    "timestamp",
    "watch_ratio",
]
KUAI_REC_USER_SPARSE_COLUMNS = [
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
KUAI_REC_USER_NUMERIC_COLUMNS = [
    "follow_user_num",
    "fans_user_num",
    "friend_user_num",
    "register_days",
]
KUAI_REC_ITEM_DAILY_SPARSE_COLUMNS = [
    "author_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "visible_status",
    "music_id",
    "video_tag_id",
    "video_tag_name",
]
KUAI_REC_ITEM_DAILY_NUMERIC_COLUMNS = [
    "video_duration_daily",
    "video_width",
    "video_height",
]
KUAI_REC_ITEM_DAILY_AGGREGATE_NUMERIC_COLUMNS = [
    "show_cnt",
    "show_user_num",
    "play_cnt",
    "play_user_num",
    "play_duration_daily",
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
    "follow_user_num_item",
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
]
KUAI_REC_MATRIX_SPARSE_COLUMNS = ["date"]
KUAI_REC_MATRIX_NUMERIC_COLUMNS = ["timestamp"]
RAW_SPARSE_PREFIX = "__raw_sparse_"
RAW_NUMERIC_PREFIX = "__raw_numeric_"
RAW_SEQUENCE_PREFIX = "__raw_seq_"
RAW_SEQUENCE_MASK_PREFIX = "__raw_seq_mask_"
KUAI_REC_CATEGORY_ID_COLUMNS = [
    "first_level_category_id",
    "second_level_category_id",
    "third_level_category_id",
]
KUAI_REC_TEXT_COLUMNS = ["caption", "manual_cover_text", "topic_tag"]
KUAI_REC_TEXT_HASH_COLUMN = "text_hash"
KUAI_REC_CONTENT_NUMERIC_NAMES = [
    "has_caption",
    "has_cover_text",
    "has_any_text",
    "log_caption_chars",
    "log_cover_text_chars",
    "log_topic_tag_chars",
]


def _watch_event_names(config: dict[str, Any]) -> list[str]:

    configured = config.get("event_taxonomy", {}).get("watch_event_names")
    if configured:
        return [str(name) for name in configured]

    return list(WATCH_EVENT_NAMES)


def _event_names(config: dict[str, Any]) -> list[str]:

    names = _watch_event_names(config)

    if bool(config.get("event_taxonomy", {}).get("factorized_exit", False)):
        session_end_name = str(config.get("event_taxonomy", {}).get("session_end_event_name", "session_end"))
        if session_end_name not in names:
            names = [*names, session_end_name]
    return names


def _unknown_prev_event(config: dict[str, Any]) -> int:

    return len(_event_names(config))


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:

    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _normalize_category_ids(series: pd.Series) -> pd.Series:

    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    unknown_values = {"", "-124", "UNKNOWN", "unknown", "nan", "NaN", "None", "<NA>"}
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


def _clean_text(series: pd.Series) -> pd.Series:

    text = series.fillna("").astype(str).str.strip()

    missing_values = {"", "-124", "UNKNOWN", "unknown", "nan", "NaN", "None", "<NA>"}
    return text.mask(text.isin(missing_values), "")


def _stable_text_hash(text: str, buckets: int) -> int:

    if not text:
        return 0

    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest, 16) % max(int(buckets), 1) + 1


def _read_caption_category_loose(path: Path, logger) -> pd.DataFrame:
    logger.info("Reading KuaiRec content metadata with malformed-row handling. path=%s", path)

    rows: list[dict[str, str]] = []
    bad_rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            parts = [part.strip() for part in line.rstrip("\n").split(",")]
            if len(parts) < 10:
                bad_rows += 1
                continue
            left = parts[:3]
            right = parts[-6:]
            middle = parts[3:-6]
            rows.append(
                {
                    "video_id": left[0],
                    "manual_cover_text": left[1],
                    "caption": left[2],
                    "topic_tag": ",".join(middle),
                    "first_level_category_id": right[0],
                    "second_level_category_id": right[2],
                    "third_level_category_id": right[4],
                }
            )

    metadata = pd.DataFrame(rows)
    metadata["video_id"] = pd.to_numeric(metadata["video_id"], errors="coerce").astype("Int64")
    metadata = metadata.dropna(subset=["video_id"]).copy()
    metadata["video_id"] = metadata["video_id"].astype(np.int64)
    logger.info("KuaiRec content metadata loaded. rows=%s bad_rows=%s", len(metadata), bad_rows)
    return metadata


def _read_matrix(config: dict[str, Any], logger) -> pd.DataFrame:
    logger.info("Reading the KuaiRec interaction matrix...")

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"]).expanduser().resolve()
    matrix_name = str(data_cfg.get("matrix", "small_matrix"))
    csv_name = f"{matrix_name}.csv"
    csv_path = raw_dir / "data" / csv_name
    zip_path = Path(data_cfg.get("zip_path", raw_dir.with_suffix(".zip"))).expanduser().resolve()
    nrows = config.get("debug", {}).get("max_rows", {}).get("standard_per_file")

    if csv_path.exists():
        frame = pd.read_csv(csv_path, usecols=MATRIX_COLUMNS, nrows=nrows)
    elif zip_path.exists():
        inner = f"KuaiRec 2.0/data/{csv_name}"
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(inner) as handle:
                frame = pd.read_csv(handle, usecols=MATRIX_COLUMNS, nrows=nrows)
    else:
        raise FileNotFoundError(f"KuaiRec matrix not found: {csv_path} or {zip_path}")

    logger.info("KuaiRec interaction matrix loaded. rows=%s matrix=%s", len(frame), matrix_name)
    return frame


def _read_item_daily_video_type(config: dict[str, Any], logger) -> pd.DataFrame:
    logger.info("Reading KuaiRec item_daily video_type...")

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"]).expanduser().resolve()
    csv_path = raw_dir / "data" / "item_daily_features.csv"
    zip_path = Path(data_cfg.get("zip_path", raw_dir.with_suffix(".zip"))).expanduser().resolve()

    usecols = ["video_id", "video_type"]
    if csv_path.exists():
        frame = pd.read_csv(csv_path, usecols=usecols)
    elif zip_path.exists():
        inner = "KuaiRec 2.0/data/item_daily_features.csv"
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(inner) as handle:
                frame = pd.read_csv(handle, usecols=usecols)
    else:
        raise FileNotFoundError(f"KuaiRec item_daily_features not found: {csv_path} or {zip_path}")

    frame = frame.drop_duplicates(subset=["video_id"], keep="first")
    logger.info("KuaiRec item_daily video_type loaded. rows=%s", len(frame))
    return frame


def _read_kuairec_csv(
    config: dict[str, Any],
    filename: str,
    usecols: list[str] | None,
    logger,
    nrows: int | None = None,
) -> pd.DataFrame:
    logger.info("Reading a KuaiRec raw feature table. file=%s", filename)

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"]).expanduser().resolve()
    csv_path = raw_dir / "data" / filename
    zip_path = Path(data_cfg.get("zip_path", raw_dir.with_suffix(".zip"))).expanduser().resolve()

    if csv_path.exists():
        frame = pd.read_csv(csv_path, usecols=usecols, nrows=nrows)
    elif zip_path.exists():
        inner = f"KuaiRec 2.0/data/{filename}"
        with zipfile.ZipFile(zip_path) as archive:
            with archive.open(inner) as handle:
                frame = pd.read_csv(handle, usecols=usecols, nrows=nrows)
    else:
        raise FileNotFoundError(f"KuaiRec raw table not found: {csv_path} or {zip_path}")

    logger.info("KuaiRec raw feature table loaded. file=%s rows=%s", filename, len(frame))
    return frame


def _normalize_sparse_value(series: pd.Series) -> pd.Series:

    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    unknown_values = {"", "nan", "NaN", "None", "<NA>", "UNKNOWN", "unknown"}
    return text.mask(text.isin(unknown_values), "")


def _encode_sparse_columns(frame: pd.DataFrame, columns: list[str], logger) -> tuple[list[str], list[int], dict[str, dict[str, int]]]:
    logger.info("Encoding KuaiRec raw sparse features. cols=%s", len(columns))

    raw_columns: list[str] = []
    cardinalities: list[int] = []
    audit: dict[str, dict[str, int]] = {}
    for idx, column in enumerate(columns):
        logger.info("Encoding a KuaiRec raw sparse column. idx=%s col=%s", idx, column)
        if column not in frame:
            encoded = np.zeros(len(frame), dtype=np.int64)
            cardinality = 1
            missing_rows = len(frame)
        elif pd.api.types.is_numeric_dtype(frame[column]):
            values = frame[column]
            non_missing = ~values.isna()
            encoded = np.zeros(len(values), dtype=np.int64)
            if bool(non_missing.any()):
                codes, uniques = pd.factorize(values[non_missing], sort=True)
                encoded[non_missing.to_numpy(copy=False)] = codes.astype(np.int64, copy=False) + 1
                cardinality = int(len(uniques) + 1)
            else:
                cardinality = 1
            missing_rows = int((~non_missing).sum())
        else:
            normalized = _normalize_sparse_value(frame[column])
            non_missing = normalized != ""
            encoded = np.zeros(len(normalized), dtype=np.int64)
            if bool(non_missing.any()):
                codes, uniques = pd.factorize(normalized[non_missing], sort=True)
                encoded[non_missing.to_numpy(copy=False)] = codes.astype(np.int64, copy=False) + 1
                cardinality = int(len(uniques) + 1)
            else:
                cardinality = 1
            missing_rows = int((normalized == "").sum())

        raw_column = f"{RAW_SPARSE_PREFIX}{idx}"
        frame[raw_column] = encoded
        raw_columns.append(raw_column)
        cardinalities.append(cardinality)
        audit[column] = {
            "cardinality": cardinality,
            "missing_rows": missing_rows,
        }
        logger.info("KuaiRec raw sparse column encoded. idx=%s col=%s cardinality=%s missing=%s", idx, column, cardinality, missing_rows)

    logger.info("KuaiRec raw sparse features encoded. cols=%s", len(raw_columns))
    return raw_columns, cardinalities, audit


def _encode_numeric_columns(frame: pd.DataFrame, columns: list[str], logger) -> tuple[list[str], dict[str, dict[str, int]]]:
    logger.info("Encoding KuaiRec raw numeric features. cols=%s", len(columns))

    raw_columns: list[str] = []
    audit: dict[str, dict[str, int]] = {}
    for idx, column in enumerate(columns):
        raw_value = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series([np.nan] * len(frame))
        missing = int(raw_value.isna().sum())
        values = raw_value.fillna(0.0).astype(np.float32).to_numpy(copy=False)
        if column != "play_progress":
            values = np.log1p(np.clip(values, 0, None)).astype(np.float32, copy=False)
        raw_column = f"{RAW_NUMERIC_PREFIX}{idx}"
        frame[raw_column] = values
        raw_columns.append(raw_column)
        audit[column] = {"missing_rows": missing}

    logger.info("KuaiRec raw numeric features encoded. cols=%s", len(raw_columns))
    return raw_columns, audit


def _parse_feature_sequence(value: object) -> list[str]:

    if value is None:
        return []
    text = str(value).strip()
    if not text or text in {"nan", "NaN", "None", "UNKNOWN"}:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = []
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def _attach_raw_features(frame: pd.DataFrame, config: dict[str, Any], logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    logger.info("Merging KuaiRec raw features...")

    user_features = _read_kuairec_csv(config, "user_features.csv", None, logger)
    frame = frame.merge(user_features, on="user_id", how="left")

    item_daily = _read_kuairec_csv(config, "item_daily_features.csv", None, logger)
    item_daily = item_daily.rename(
        columns={
            "video_duration": "video_duration_daily",
            "play_duration": "play_duration_daily",
            "follow_user_num": "follow_user_num_item",
        }
    )
    frame = frame.merge(item_daily, on=["video_id", "date"], how="left")

    item_categories = _read_kuairec_csv(config, "item_categories.csv", ["video_id", "feat"], logger)
    sequence_length = int(config.get("raw_features", {}).get("sequence_length", 8))
    category_sequences = item_categories["feat"].map(_parse_feature_sequence)
    token_values = sorted({token for values in category_sequences for token in values})
    token_mapping = {token: idx + 1 for idx, token in enumerate(token_values)}
    sequence_columns = [f"{RAW_SEQUENCE_PREFIX}{pos}" for pos in range(sequence_length)]
    sequence_mask_columns = [f"{RAW_SEQUENCE_MASK_PREFIX}{pos}" for pos in range(sequence_length)]
    for pos, value_column in enumerate(sequence_columns):
        mask_column = sequence_mask_columns[pos]
        item_categories[value_column] = category_sequences.map(
            lambda values, pos=pos: token_mapping.get(values[pos], 0) if pos < len(values) else 0
        ).astype(np.int64)
        item_categories[mask_column] = (item_categories[value_column].to_numpy(copy=False) > 0).astype(np.float32)
    sequence_non_empty = int(category_sequences.map(bool).sum())
    frame = frame.merge(item_categories[["video_id", *sequence_columns, *sequence_mask_columns]], on="video_id", how="left")
    for value_column in sequence_columns:
        frame[value_column] = frame[value_column].fillna(0).astype(np.int64)
    for mask_column in sequence_mask_columns:
        frame[mask_column] = frame[mask_column].fillna(0.0).astype(np.float32)

    include_aggregate_statistics = bool(config.get("raw_features", {}).get("include_aggregate_statistics", False))
    sparse_names = [
        *[f"user::{column}" for column in KUAI_REC_USER_SPARSE_COLUMNS],
        *[f"item_daily::{column}" for column in KUAI_REC_ITEM_DAILY_SPARSE_COLUMNS],
        *[f"matrix::{column}" for column in KUAI_REC_MATRIX_SPARSE_COLUMNS],
    ]
    sparse_columns = [
        *KUAI_REC_USER_SPARSE_COLUMNS,
        *KUAI_REC_ITEM_DAILY_SPARSE_COLUMNS,
        *KUAI_REC_MATRIX_SPARSE_COLUMNS,
    ]
    raw_sparse_columns, raw_sparse_cardinalities, sparse_audit = _encode_sparse_columns(frame, sparse_columns, logger)

    item_daily_numeric_columns = [
        *KUAI_REC_ITEM_DAILY_NUMERIC_COLUMNS,
        *(KUAI_REC_ITEM_DAILY_AGGREGATE_NUMERIC_COLUMNS if include_aggregate_statistics else []),
    ]
    numeric_names = [
        *[f"user::{column}" for column in KUAI_REC_USER_NUMERIC_COLUMNS],
        *[f"item_daily::{column}" for column in item_daily_numeric_columns],
        *[f"matrix::{column}" for column in KUAI_REC_MATRIX_NUMERIC_COLUMNS],
    ]
    numeric_columns = [
        *KUAI_REC_USER_NUMERIC_COLUMNS,
        *item_daily_numeric_columns,
        *KUAI_REC_MATRIX_NUMERIC_COLUMNS,
    ]
    raw_numeric_columns, numeric_audit = _encode_numeric_columns(frame, numeric_columns, logger)

    meta = {
        "enabled": True,
        "raw_sparse_columns": raw_sparse_columns,
        "raw_sparse_feature_names": sparse_names,
        "raw_sparse_cardinalities": [int(value) for value in raw_sparse_cardinalities],
        "num_raw_sparse_features": int(len(raw_sparse_columns)),
        "raw_numeric_columns": raw_numeric_columns,
        "raw_numeric_feature_names": numeric_names,
        "num_raw_numeric_features": int(len(raw_numeric_columns)),
        "raw_sequence_columns": sequence_columns,
        "raw_sequence_mask_columns": sequence_mask_columns,
        "raw_sequence_feature_name": "item_categories::feat",
        "raw_sequence_cardinality": int(len(token_mapping) + 1),
        "raw_sequence_length": int(sequence_length),
        "sparse_audit": sparse_audit,
        "numeric_audit": numeric_audit,
        "sequence_audit": {
            "non_empty_rows": sequence_non_empty,
            "token_cardinality": int(len(token_mapping) + 1),
        },
        "include_aggregate_statistics": include_aggregate_statistics,
    }
    logger.info(
        "KuaiRec raw features merged. sparse=%s numeric=%s seq_len=%s",
        meta["num_raw_sparse_features"],
        meta["num_raw_numeric_features"],
        meta["raw_sequence_length"],
    )
    return frame, meta


def _is_row_shuffle_split(config: dict[str, Any]) -> bool:

    unit = str(config.get("split", {}).get("unit", "row_shuffle_80_10_10"))
    return unit in {"row_shuffle", "row_shuffle_80_10_10"}


def _apply_row_shuffle_protocol_filters(frame: pd.DataFrame, config: dict[str, Any], logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    logger.info("Applying the KuaiRec row-level random-protocol filter...")

    before_rows = int(len(frame))
    item_daily = _read_item_daily_video_type(config, logger)
    frame = frame.merge(item_daily, on="video_id", how="left")
    ad_mask = frame["video_type"].fillna("12345").astype(str) == "AD"
    frame = frame.loc[~ad_mask].copy()
    frame.drop(columns=["video_type"], inplace=True)

    after_ad_rows = int(len(frame))
    positive_mask = (frame["play_duration"] > 0) & (frame["video_duration"] > 0)
    frame = frame.loc[positive_mask].copy()

    after_positive_rows = int(len(frame))
    bounded_mask = frame["play_duration"] < frame["video_duration"] * 10
    frame = frame.loc[bounded_mask].reset_index(drop=True)

    audit = {
        "protocol": "row_shuffle_80_10_10",
        "rows_before_filters": before_rows,
        "ad_video_rows_removed": int(ad_mask.sum()),
        "rows_after_ad_filter": after_ad_rows,
        "non_positive_duration_rows_removed": int(after_ad_rows - after_positive_rows),
        "over_10x_duration_rows_removed": int((~bounded_mask).sum()),
        "rows_after_filters": int(len(frame)),
        "split_shuffle_random_state": 1234,
    }
    logger.info("KuaiRec row-level random-protocol filtering complete. audit=%s", audit)
    return frame, audit


def _sanitize_and_construct(frame: pd.DataFrame, config: dict[str, Any], upper_bounds: np.ndarray, logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    logger.info("Building KuaiRec event labels and session features...")

    taxonomy = config["event_taxonomy"]
    prep = config["preprocessing"]
    min_duration_ms = int(prep.get("min_duration_ms", 1))
    invalid_duration = frame["video_duration"] <= 0
    if str(prep.get("invalid_duration_policy", "drop")) == "drop":
        frame = frame.loc[~invalid_duration].copy()
    elif invalid_duration.any():
        raise ValueError(f"Found invalid video_duration rows: {int(invalid_duration.sum())}")
    frame["duration_ms_safe"] = frame["video_duration"].clip(lower=min_duration_ms)
    frame["play_duration_safe"] = frame["play_duration"].clip(lower=0)

    missing_timestamp = frame["timestamp"].isna()
    if missing_timestamp.any():
        order = frame.groupby("user_id", sort=False).cumcount().astype(np.float64)
        user_offset = pd.factorize(frame["user_id"], sort=True)[0].astype(np.float64) * 10_000_000.0
        frame.loc[missing_timestamp, "timestamp"] = (user_offset + order)[missing_timestamp]

    alpha = float(taxonomy.get("alpha", 0.3))
    beta = float(taxonomy.get("beta", 1.1))
    mid_cut = float(taxonomy.get("mid_completion_cut", 0.9))
    ratio = frame["play_duration_safe"].to_numpy(np.float64) / frame["duration_ms_safe"].to_numpy(np.float64)
    watch_event = np.zeros(len(frame), dtype=np.int64)
    watch_event[(ratio >= alpha) & (ratio < mid_cut)] = 1
    watch_event[(ratio >= mid_cut) & (ratio <= beta)] = 2
    watch_event[ratio > beta] = 3
    frame["watch_event"] = watch_event
    frame["event"] = watch_event

    frame.sort_values(["user_id", "timestamp", "video_id"], inplace=True, kind="mergesort")
    frame.reset_index(drop=True, inplace=True)
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
    frame["prev_play_sec"] = (session_group["play_duration_safe"].shift(1).fillna(0).clip(lower=0) / 1000.0).astype(np.float32)
    unknown_prev_event = _unknown_prev_event(config)
    frame["prev_event"] = session_group["event"].shift(1).fillna(unknown_prev_event).astype(np.int64)
    is_session_end = next_ts.isna() | (next_gap_sec > gamma_sec)
    frame["is_session_end"] = is_session_end.astype(np.float32)

    frame["time_bin"] = assign_time_bins(frame["play_duration_safe"].to_numpy(np.float64) / 1000.0, upper_bounds)
    frame["duration_sec"] = (frame["duration_ms_safe"] / 1000.0).astype(np.float32)
    frame["play_time_sec"] = (frame["play_duration_safe"] / 1000.0).astype(np.float32)
    frame["log_duration"] = np.log1p(frame["duration_sec"]).astype(np.float32)
    hour = ((frame["timestamp"].to_numpy(np.float64) / 3600.0) % 24.0).astype(np.float32)
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    frame["tab"] = 0
    frame["engagement_mark"] = 0.0

    watch_event_names = _watch_event_names(config)
    watch_counts = pd.Series(frame["watch_event"]).value_counts().reindex(range(len(watch_event_names)), fill_value=0)
    audit = {
        "rows_after_duration_policy": int(len(frame)),
        "invalid_duration_rows": int(invalid_duration.sum()),
        "missing_timestamp_rows_filled": int(missing_timestamp.sum()),
        "sessions": int(frame["session_key"].nunique()),
        "watch_event_counts": {watch_event_names[idx]: int(value) for idx, value in watch_counts.items()},
        "session_end_count": int(frame["is_session_end"].sum()),
    }
    logger.info("KuaiRec event labels and session features built. sessions=%s", audit["sessions"])
    return frame, audit


def _encode_ids(frame: pd.DataFrame, logger) -> dict[str, int]:
    logger.info("Encoding KuaiRec user and video IDs...")

    frame["user_idx"] = pd.factorize(frame["user_id"], sort=True)[0].astype(np.int64)
    frame["item_idx"] = pd.factorize(frame["video_id"], sort=True)[0].astype(np.int64)

    cardinality = {
        "num_users": int(frame["user_idx"].max()) + 1,
        "num_items": int(frame["item_idx"].max()) + 1,
    }
    logger.info("KuaiRec user and video IDs encoded. users=%s items=%s", cardinality["num_users"], cardinality["num_items"])
    return cardinality


def _assign_splits(frame: pd.DataFrame, ratios: list[float], logger) -> pd.Series:
    logger.info("Creating the standard KuaiRec split...")

    session_info = (
        frame.groupby("session_key", sort=False)
        .agg(user_id=("user_id", "first"), session_start=("timestamp", "min"))
        .reset_index()
        .sort_values(["user_id", "session_start", "session_key"], kind="mergesort")
    )
    session_info["rank"] = session_info.groupby("user_id", sort=False).cumcount()
    session_info["count"] = session_info.groupby("user_id", sort=False)["session_key"].transform("count")
    frac = (session_info["rank"] + 0.5) / session_info["count"].clip(lower=1)
    train_cut, val_cut = float(ratios[0]), float(ratios[0] + ratios[1])
    session_info["split"] = np.select([frac < train_cut, frac < val_cut], ["train", "val"], default="test")
    mapping = dict(zip(session_info["session_key"], session_info["split"]))

    split = frame["session_key"].map(mapping)
    logger.info("Standard KuaiRec split created.")
    return split


def _assign_row_shuffle_splits(frame: pd.DataFrame, ratios: list[float], logger) -> pd.Series:
    logger.info("Creating the KuaiRec row-level random split...")

    shuffled_index = frame.sample(frac=1.0, random_state=1234).index.to_numpy()
    n_rows = int(len(shuffled_index))
    n_train = int(float(ratios[0]) * n_rows)
    n_val_end = int(float(ratios[0] + ratios[1]) * n_rows)

    split = pd.Series("test", index=frame.index, dtype=object)
    split.loc[shuffled_index[:n_train]] = "train"
    split.loc[shuffled_index[n_train:n_val_end]] = "val"
    split.loc[shuffled_index[n_val_end:]] = "test"
    logger.info(
        "KuaiRec row-level random split created. train=%s val=%s test=%s",
        int((split == "train").sum()),
        int((split == "val").sum()),
        int((split == "test").sum()),
    )
    return split


def _matrix(frame: pd.DataFrame, cols: list[str], mask: np.ndarray) -> np.ndarray:

    columns: list[np.ndarray] = []
    for col in cols:
        value = frame[col].to_numpy(copy=False)[mask].astype(np.float32, copy=False)
        if col in {"session_pos", "elapsed_session_sec", "prev_gap_sec", "prev_play_sec"}:
            value = np.log1p(np.clip(value, 0, None)).astype(np.float32, copy=False)
        columns.append(value)
    return np.column_stack(columns).astype(np.float32, copy=False)


def _arrays_for_mask(
    frame: pd.DataFrame,
    mask: np.ndarray,
    raw_meta: dict[str, Any] | None = None,
    unknown_prev_event: int = UNKNOWN_PREV_EVENT,
) -> dict[str, np.ndarray]:

    arrays = {
        "user_idx": frame["user_idx"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "item_idx": frame["item_idx"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "tab": frame["tab"].to_numpy(copy=False)[mask].astype(np.int64, copy=False),
        "prev_event": np.clip(frame["prev_event"].to_numpy(copy=False)[mask], 0, int(unknown_prev_event)).astype(np.int64, copy=False),
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

    if raw_meta and raw_meta.get("enabled", False):
        raw_sparse_columns = list(raw_meta.get("raw_sparse_columns", []))
        raw_numeric_columns = list(raw_meta.get("raw_numeric_columns", []))
        raw_sequence_columns = list(raw_meta.get("raw_sequence_columns", []))
        raw_sequence_mask_columns = list(raw_meta.get("raw_sequence_mask_columns", []))
        arrays["raw_sparse"] = (
            frame[raw_sparse_columns].to_numpy(np.int64, copy=False)[mask]
            if raw_sparse_columns
            else np.zeros((int(mask.sum()), 0), dtype=np.int64)
        )
        arrays["raw_numeric"] = (
            frame[raw_numeric_columns].to_numpy(np.float32, copy=False)[mask]
            if raw_numeric_columns
            else np.zeros((int(mask.sum()), 0), dtype=np.float32)
        )
        arrays["raw_seq"] = (
            frame[raw_sequence_columns].to_numpy(np.int64, copy=False)[mask]
            if raw_sequence_columns
            else np.zeros((int(mask.sum()), 0), dtype=np.int64)
        )
        arrays["raw_seq_mask"] = (
            frame[raw_sequence_mask_columns].to_numpy(np.float32, copy=False)[mask]
            if raw_sequence_mask_columns
            else np.zeros((int(mask.sum()), 0), dtype=np.float32)
        )
    else:
        arrays["raw_sparse"] = np.zeros((int(mask.sum()), 0), dtype=np.int64)
        arrays["raw_numeric"] = np.zeros((int(mask.sum()), 0), dtype=np.float32)
        arrays["raw_seq"] = np.zeros((int(mask.sum()), 0), dtype=np.int64)
        arrays["raw_seq_mask"] = np.zeros((int(mask.sum()), 0), dtype=np.float32)

    return arrays


def _build_item_content_features(
    frame: pd.DataFrame,
    cardinality: dict[str, int],
    config: dict[str, Any],
    output_dir: Path,
    logger,
) -> dict[str, Any]:
    logger.info("Building KuaiRec content features...")

    content_cfg = config.get("content_features", {})
    num_items = int(cardinality["num_items"])
    item_content_path = output_dir / "item_content.npz"
    if not bool(content_cfg.get("enabled", False)):
        np.savez(
            item_content_path,
            content_category=np.zeros((num_items, 0), dtype=np.int64),
            content_numeric=np.zeros((num_items, 0), dtype=np.float32),
        )
        logger.info("KuaiRec content features disabled; writing empty item_content.")
        return {
            "enabled": False,
            "num_content_category_levels": 0,
            "content_category_cardinalities": [],
            "num_content_numeric_features": 0,
            "content_numeric_names": [],
            "item_content_path": str(item_content_path),
        }

    metadata_path = Path(
        content_cfg.get(
            "caption_category_path",
            Path(config["data"]["raw_dir"]).expanduser().resolve() / "kuairec_caption_category.csv",
        )
    ).expanduser().resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"KuaiRec content metadata not found: {metadata_path}")
    metadata = _read_caption_category_loose(metadata_path, logger)

    item_lookup = (
        frame[["video_id", "item_idx"]]
        .drop_duplicates("item_idx")
        .sort_values("item_idx", kind="mergesort")
    )
    aligned = item_lookup.merge(metadata, on="video_id", how="left").sort_values("item_idx", kind="mergesort")
    if len(aligned) != num_items:
        raise ValueError(f"KuaiRec content alignment failed: rows={len(aligned)} items={num_items}")

    category_arrays: list[np.ndarray] = []
    category_cardinalities: list[int] = []
    category_audit: dict[str, dict[str, int]] = {}
    for column in KUAI_REC_CATEGORY_ID_COLUMNS:
        encoded, mapping, audit = _encode_category_level(aligned[column])
        category_arrays.append(encoded)
        category_cardinalities.append(len(mapping) + 1)
        category_audit[column] = audit

    include_text_features = bool(content_cfg.get("include_text_features", False))
    content_numeric_names: list[str] = []
    text_audit: dict[str, Any] = {}
    if include_text_features:
        logger.info("Building legacy KuaiRec text content features.")

        caption_text = _clean_text(aligned["caption"])
        cover_text = _clean_text(aligned["manual_cover_text"])
        topic_text = _clean_text(aligned["topic_tag"])
        hash_text = (caption_text + " " + cover_text).str.strip()

        text_hash_buckets = int(content_cfg.get("text_hash_buckets", 4096))
        text_hash = hash_text.map(lambda value: _stable_text_hash(value, text_hash_buckets)).astype(np.int64).to_numpy()
        category_arrays.append(text_hash)
        category_cardinalities.append(text_hash_buckets + 1)
        category_audit[KUAI_REC_TEXT_HASH_COLUMN] = {
            "hash_buckets": int(text_hash_buckets),
            "missing_text_rows": int((hash_text == "").sum()),
        }

        has_caption = (caption_text != "").astype(np.float32).to_numpy()
        has_cover_text = (cover_text != "").astype(np.float32).to_numpy()
        has_any_text = (hash_text != "").astype(np.float32).to_numpy()
        log_caption_chars = np.log1p(caption_text.str.len().to_numpy(np.float32)).astype(np.float32, copy=False)
        log_cover_text_chars = np.log1p(cover_text.str.len().to_numpy(np.float32)).astype(np.float32, copy=False)
        log_topic_tag_chars = np.log1p(topic_text.str.len().to_numpy(np.float32)).astype(np.float32, copy=False)
        content_numeric = np.column_stack(
            [
                has_caption,
                has_cover_text,
                has_any_text,
                log_caption_chars,
                log_cover_text_chars,
                log_topic_tag_chars,
            ]
        ).astype(np.float32, copy=False)
        content_numeric_names = [*KUAI_REC_CONTENT_NUMERIC_NAMES]
        text_audit = {
            "caption_non_empty_rows": int(has_caption.sum()),
            "cover_text_non_empty_rows": int(has_cover_text.sum()),
            "any_text_non_empty_rows": int(has_any_text.sum()),
        }
        logger.info("Legacy KuaiRec text content features built. text_hash_buckets=%s", text_hash_buckets)
    else:
        content_numeric = np.zeros((num_items, 0), dtype=np.float32)
        content_numeric_names = []

    content_category = np.column_stack(category_arrays).astype(np.int64, copy=False)
    np.savez(item_content_path, content_category=content_category, content_numeric=content_numeric)

    meta = {
        "enabled": True,
        "caption_category_path": str(metadata_path),
        "num_content_category_levels": int(content_category.shape[1]),
        "content_category_names": [
            *KUAI_REC_CATEGORY_ID_COLUMNS,
            *([KUAI_REC_TEXT_HASH_COLUMN] if include_text_features else []),
        ],
        "content_category_cardinalities": [int(value) for value in category_cardinalities],
        "num_content_numeric_features": int(content_numeric.shape[1]),
        "content_numeric_names": content_numeric_names,
        "item_content_path": str(item_content_path),
        "category_audit": category_audit,
    }
    if include_text_features:
        meta["text_audit"] = text_audit
    logger.info(
        "KuaiRec content features built. category_levels=%s numeric=%s",
        meta["num_content_category_levels"],
        meta["num_content_numeric_features"],
    )
    return meta


def _save_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def prepare_kuairec(config: dict[str, Any], logger) -> dict[str, Any]:
    logger.info("Preprocessing the KuaiRec benchmark...")

    output_dir = Path(config["data"]["processed_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    upper_bounds = build_time_bin_upper_bounds(
        config["time_bins"]["linear_seconds"],
        config["time_bins"]["max_seconds"],
        config["time_bins"]["exponential_growth"],
    )
    frame = _read_matrix(config, logger)
    row_shuffle_audit: dict[str, Any] | None = None
    if _is_row_shuffle_split(config):
        frame, row_shuffle_audit = _apply_row_shuffle_protocol_filters(frame, config, logger)
    frame, event_audit = _sanitize_and_construct(frame, config, upper_bounds, logger)
    frame, raw_feature_meta = _attach_raw_features(frame, config, logger)
    cardinality = _encode_ids(frame, logger)
    event_names = _event_names(config)
    watch_event_names = _watch_event_names(config)
    unknown_prev_event = _unknown_prev_event(config)
    if _is_row_shuffle_split(config):
        frame["split"] = _assign_row_shuffle_splits(frame, config["split"]["ratios"], logger)
    else:
        frame["split"] = _assign_splits(frame, config["split"]["ratios"], logger)

    content_meta = _build_item_content_features(frame, cardinality, config, output_dir, logger)

    compressed = bool(config["data"].get("compressed_npz", False))
    split_counts: dict[str, int] = {}
    empty_mask = np.zeros(len(frame), dtype=bool)
    for split_name in ["train", "val", "test"]:
        mask = frame["split"].to_numpy(copy=False) == split_name
        split_counts[f"standard_{split_name}"] = int(mask.sum())
        arrays = _arrays_for_mask(frame, mask, raw_feature_meta, unknown_prev_event)
        _save_npz(output_dir / f"standard_{split_name}.npz", arrays, compressed)
        del arrays
        empty_arrays = _arrays_for_mask(frame, empty_mask, raw_feature_meta, unknown_prev_event)
        split_counts[f"random_{split_name}"] = 0
        _save_npz(output_dir / f"random_{split_name}.npz", empty_arrays, compressed)
        del empty_arrays

    manifest = {
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
        "num_tabs": 1,
        "num_numeric_features": len(NUMERIC_NAMES),
        "numeric_feature_names": NUMERIC_NAMES,
        "num_session_numeric_features": len(SESSION_NUMERIC_NAMES),
        "session_numeric_feature_names": SESSION_NUMERIC_NAMES,
        "content_features": content_meta,
        "raw_features": raw_feature_meta,
        "standard_audit": event_audit,
        "random_audit": {"rows_after_duration_policy": 0},
        "split_counts": split_counts,
        "split_policy": config["split"],
        "row_shuffle_protocol_audit": row_shuffle_audit,
        "event_taxonomy": config["event_taxonomy"],
        "preprocessing": config["preprocessing"],
    }
    manifest = portableize_serialized_paths(manifest)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    logger.info("KuaiRec benchmark preprocessing complete. split_counts=%s", split_counts)
    return manifest
