#!/usr/bin/env python3
"""
Add session-wise minute affect labels to an EEG CSV.

This version is designed for datasets with:
- user_id
- session_id
- timestamp as Unix seconds, e.g. 1763417280.4524
- existing arousal/valence columns that should NOT be overwritten

It can:
1. Generate a template with one row per user/session/minute.
2. Merge a filled annotation template back into the EEG rows.

Recommended usage:

Generate template:
    python add_minute_affect.py \
        --eeg-csv dataset.csv \
        --generate-template affect_template.csv

Merge filled template:
    python add_minute_affect.py \
        --eeg-csv dataset.csv \
        --annotations affect_template_filled.csv \
        --out labeled_dataset.csv

The filled annotation CSV should contain:
    user_id,session_id,affect_minute,manual_valence,manual_arousal

or:
    user_id,session_id,minute,valence,arousal

The script will output:
    session_elapsed_seconds
    affect_minute
    manual_valence
    manual_arousal
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Optional, Tuple, List

import pandas as pd


# ---------------------------------------------------------------------
# Time detection
# ---------------------------------------------------------------------

def detect_time_column(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect time-related columns.

    Returns:
        ("timestamp", col) for Unix timestamps or datetime strings
        ("seconds", col) for elapsed seconds
        ("sample", col) for sample index
    """
    timestamp_candidates = [
        "timestamp",
        "Timestamp",
        "time_stamp",
        "TimeStamp",
        "datetime",
        "DateTime",
        "date",
        "Date",
    ]

    for col in timestamp_candidates:
        if col in df.columns:
            return "timestamp", col

    seconds_candidates = [
        "seconds",
        "Seconds",
        "sec",
        "Sec",
        "elapsed",
        "Elapsed",
        "time",
        "Time",
        "time_elapsed",
        "TimeElapsed",
    ]

    for col in seconds_candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return "seconds", col

    sample_candidates = [
        "sample",
        "Sample",
        "sample_index",
        "SampleIndex",
        "index",
        "Index",
    ]

    for col in sample_candidates:
        if col in df.columns and pd.api.types.is_integer_dtype(df[col]):
            return "sample", col

    raise ValueError(
        "Could not detect a time column. Expected one of: timestamp, seconds, "
        "time_elapsed, sample, or provide --time-col explicitly."
    )


def numeric_timestamp_to_seconds(series: pd.Series) -> pd.Series:
    """
    Convert numeric timestamp-like values into seconds.

    Handles:
        Unix seconds:      ~1.7e9
        Unix milliseconds: ~1.7e12
        Unix microseconds: ~1.7e15
        Unix nanoseconds:  ~1.7e18

    If values are small, treats them as already being seconds.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)

    non_na = s.dropna()
    if non_na.empty:
        raise ValueError("Timestamp column is entirely empty or non-numeric.")

    median_abs = non_na.abs().median()

    if median_abs >= 1e17:
        # Unix nanoseconds
        return s / 1e9
    elif median_abs >= 1e14:
        # Unix microseconds
        return s / 1e6
    elif median_abs >= 1e11:
        # Unix milliseconds
        return s / 1e3
    else:
        # Unix seconds or already-relative seconds
        return s


def groupwise_min(series: pd.Series, df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    """
    Compute minimum within each group, preserving original row order.
    If group_cols is empty, compute global minimum.
    """
    if not group_cols:
        return pd.Series(series.min(), index=series.index)

    return series.groupby([df[c] for c in group_cols], sort=False).transform("min")


def compute_elapsed_seconds(
    df: pd.DataFrame,
    mode: str,
    col: str,
    group_cols: List[str],
    sample_rate: Optional[float] = None,
) -> pd.Series:
    """
    Compute elapsed seconds within each user/session group.
    """
    if col not in df.columns:
        raise ValueError(f"Time column '{col}' not found in CSV.")

    if mode == "timestamp":
        raw = df[col]

        if pd.api.types.is_numeric_dtype(raw):
            seconds = numeric_timestamp_to_seconds(raw)
            start_seconds = groupwise_min(seconds, df, group_cols)
            return seconds - start_seconds

        parsed = pd.to_datetime(raw, errors="coerce")

        if parsed.isna().all():
            raise ValueError(f"Column '{col}' could not be parsed as datetimes.")

        start_times = groupwise_min(parsed, df, group_cols)
        return (parsed - start_times).dt.total_seconds()

    if mode == "seconds":
        seconds = pd.to_numeric(df[col], errors="coerce").astype(float)
        start_seconds = groupwise_min(seconds, df, group_cols)
        return seconds - start_seconds

    if mode == "sample":
        if sample_rate is None or sample_rate <= 0:
            raise ValueError("sample_rate must be provided and > 0 when using sample mode.")

        samples = pd.to_numeric(df[col], errors="coerce").astype(float)
        start_samples = groupwise_min(samples, df, group_cols)
        return (samples - start_samples) / float(sample_rate)

    raise ValueError(f"Unsupported time mode: {mode}")


# ---------------------------------------------------------------------
# Group handling
# ---------------------------------------------------------------------

def resolve_group_cols(df: pd.DataFrame, group_cols_arg: str, no_grouping: bool) -> List[str]:
    """
    Resolve grouping columns.

    Default behavior:
        Use user_id and session_id if present.
        Otherwise use session_id if present.
        Otherwise no grouping.
    """
    if no_grouping:
        return []

    if group_cols_arg.strip().lower() == "auto":
        preferred = ["user_id", "session_id"]
        found = [c for c in preferred if c in df.columns]

        if found:
            return found

        if "session_id" in df.columns:
            return ["session_id"]

        return []

    if not group_cols_arg.strip():
        return []

    group_cols = [c.strip() for c in group_cols_arg.split(",") if c.strip()]

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Grouping columns not found in CSV: {missing}")

    return group_cols


# ---------------------------------------------------------------------
# Minute computation
# ---------------------------------------------------------------------

def build_minute_index(elapsed_seconds: pd.Series) -> pd.Series:
    """
    Compute affect_minute = floor(session_elapsed_seconds / 60).
    """
    minute = elapsed_seconds // 60
    return minute.fillna(-1).astype(int)


def add_time_columns(
    df: pd.DataFrame,
    elapsed_seconds: pd.Series,
    minute_index: pd.Series,
) -> pd.DataFrame:
    """
    Add session_elapsed_seconds and affect_minute to a copy of the dataframe.
    """
    out = df.copy()

    if "session_elapsed_seconds" in out.columns:
        out = out.drop(columns=["session_elapsed_seconds"])

    if "affect_minute" in out.columns:
        out = out.drop(columns=["affect_minute"])

    out["session_elapsed_seconds"] = elapsed_seconds.values
    out["affect_minute"] = minute_index.values

    return out


# ---------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------

def generate_template(
    df: pd.DataFrame,
    elapsed_seconds: pd.Series,
    group_cols: List[str],
    out_path: str,
    valence_col: str,
    arousal_col: str,
) -> None:
    """
    Generate one row per group/minute for manual annotation.
    """
    work = df[group_cols].copy() if group_cols else pd.DataFrame(index=df.index)
    work["session_elapsed_seconds"] = elapsed_seconds.values

    records = []

    if group_cols:
        grouped = work.groupby(group_cols, sort=False, dropna=False)

        for key, group in grouped:
            if not isinstance(key, tuple):
                key = (key,)

            group_values = dict(zip(group_cols, key))
            max_elapsed = group["session_elapsed_seconds"].max()

            if pd.isna(max_elapsed):
                continue

            last_minute = int(math.floor(float(max_elapsed) / 60.0))

            for minute in range(last_minute + 1):
                start_sec = minute * 60.0
                end_sec = min((minute + 1) * 60.0, float(max_elapsed))

                record = {
                    **group_values,
                    "affect_minute": minute,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    valence_col: "",
                    arousal_col: "",
                }
                records.append(record)

    else:
        max_elapsed = work["session_elapsed_seconds"].max()

        if pd.isna(max_elapsed):
            raise ValueError("Could not compute recording duration.")

        last_minute = int(math.floor(float(max_elapsed) / 60.0))

        for minute in range(last_minute + 1):
            start_sec = minute * 60.0
            end_sec = min((minute + 1) * 60.0, float(max_elapsed))

            records.append({
                "affect_minute": minute,
                "start_sec": start_sec,
                "end_sec": end_sec,
                valence_col: "",
                arousal_col: "",
            })

    template = pd.DataFrame.from_records(records)
    template.to_csv(out_path, index=False)

    print(f"Template written: {out_path}")
    print(f"Template rows: {len(template)}")


# ---------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------

def read_annotations_file(path: str) -> pd.DataFrame:
    """
    Read CSV, TSV, JSON, or JSONL annotations.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")

    if ext == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame.from_records(records)

    if ext == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)

    raise ValueError("Unsupported annotation format. Use .csv, .tsv, .jsonl, or .json")


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find the first matching column, case-insensitive.
    """
    exact = set(df.columns)
    lower_to_actual = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in exact:
            return candidate

        lowered = candidate.lower()
        if lowered in lower_to_actual:
            return lower_to_actual[lowered]

    return None


def load_annotations(
    annotations_path: str,
    group_cols: List[str],
    output_valence_col: str,
    output_arousal_col: str,
) -> pd.DataFrame:
    """
    Load annotations and normalize to:
        group_cols + affect_minute + output_valence_col + output_arousal_col

    Supports either:
        affect_minute
        minute

    Supports labels named either:
        manual_valence/manual_arousal
        minute_valence/minute_arousal
        valence/arousal
    """
    ann = read_annotations_file(annotations_path)
    ann.columns = [str(c).strip() for c in ann.columns]

    minute_col = find_column(ann, ["affect_minute", "minute"])
    start_minute_col = find_column(ann, ["start_minute", "start_affect_minute"])
    end_minute_col = find_column(ann, ["end_minute", "end_affect_minute"])

    valence_col = find_column(
        ann,
        [output_valence_col, "manual_valence", "minute_valence", "valence"],
    )
    arousal_col = find_column(
        ann,
        [output_arousal_col, "manual_arousal", "minute_arousal", "arousal"],
    )

    if valence_col is None:
        raise ValueError(
            "Could not find a valence annotation column. Expected one of: "
            f"{output_valence_col}, manual_valence, minute_valence, valence."
        )

    if arousal_col is None:
        raise ValueError(
            "Could not find an arousal annotation column. Expected one of: "
            f"{output_arousal_col}, manual_arousal, minute_arousal, arousal."
        )

    has_single_minute = minute_col is not None
    has_minute_range = start_minute_col is not None and end_minute_col is not None

    if not has_single_minute and not has_minute_range:
        raise ValueError(
            "Annotations must include either affect_minute/minute, "
            "or start_minute and end_minute."
        )

    records = []

    for _, row in ann.iterrows():
        base = {}

        for group_col in group_cols:
            if group_col in ann.columns:
                base[group_col] = row[group_col]

        label_values = {
            output_valence_col: row[valence_col],
            output_arousal_col: row[arousal_col],
        }

        if has_single_minute:
            if pd.isna(row[minute_col]):
                continue

            records.append({
                **base,
                "affect_minute": int(row[minute_col]),
                **label_values,
            })

        else:
            if pd.isna(row[start_minute_col]) or pd.isna(row[end_minute_col]):
                continue

            start_minute = int(row[start_minute_col])
            end_minute = int(row[end_minute_col])

            for minute in range(start_minute, end_minute + 1):
                records.append({
                    **base,
                    "affect_minute": minute,
                    **label_values,
                })

    out = pd.DataFrame.from_records(records)

    if out.empty:
        raise ValueError("No valid annotations found.")

    return out


# ---------------------------------------------------------------------
# Annotation merge
# ---------------------------------------------------------------------

def ensure_annotation_group_cols(
    ann: pd.DataFrame,
    df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    If the original data has only one group, allow annotations without group columns.
    If there are multiple groups, require group columns in the annotations.
    """
    if not group_cols:
        return ann

    missing = [c for c in group_cols if c not in ann.columns]

    if not missing:
        return ann

    unique_groups = df[group_cols].drop_duplicates()

    if len(unique_groups) == 1:
        fixed = ann.copy()

        for c in missing:
            fixed[c] = unique_groups.iloc[0][c]

        return fixed

    raise ValueError(
        "Annotations are missing grouping columns "
        f"{missing}. Because your CSV has multiple sessions/users, the annotation "
        "file must include the grouping columns. Usually this means using the "
        "generated template directly."
    )


def merge_annotations(
    df: pd.DataFrame,
    ann: pd.DataFrame,
    group_cols: List[str],
    output_valence_col: str,
    output_arousal_col: str,
) -> pd.DataFrame:
    """
    Merge annotations onto EEG rows using:
        group_cols + affect_minute
    """
    merge_keys = group_cols + ["affect_minute"]

    missing_df_keys = [c for c in merge_keys if c not in df.columns]
    if missing_df_keys:
        raise ValueError(f"Dataframe missing merge keys: {missing_df_keys}")

    missing_ann_keys = [c for c in merge_keys if c not in ann.columns]
    if missing_ann_keys:
        raise ValueError(f"Annotations missing merge keys: {missing_ann_keys}")

    label_cols = [output_valence_col, output_arousal_col]

    ann_small = ann[merge_keys + label_cols].copy()
    ann_small = ann_small.drop_duplicates(subset=merge_keys, keep="last")

    # If rerunning the script, avoid duplicate manual label columns.
    df_clean = df.copy()
    for label_col in label_cols:
        if label_col in df_clean.columns:
            df_clean = df_clean.drop(columns=[label_col])

    merged = df_clean.merge(
        ann_small,
        how="left",
        on=merge_keys,
    )

    return merged


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add session-wise minute affect labels to an EEG CSV."
    )

    parser.add_argument(
        "--eeg-csv",
        "--eego-csv",
        dest="eeg_csv",
        required=True,
        help="Path to EEG CSV file.",
    )

    parser.add_argument(
        "--out",
        help="Output CSV path. Default: <input>_with_affect.csv",
    )

    parser.add_argument(
        "--generate-template",
        help="Path to write the annotation template CSV.",
    )

    parser.add_argument(
        "--annotations",
        help="Path to filled annotations file: CSV, TSV, JSONL, or JSON.",
    )

    parser.add_argument(
        "--sample-rate",
        type=float,
        help="Sample rate in Hz, required only when using sample index time mode.",
    )

    parser.add_argument(
        "--time-mode",
        choices=["auto", "timestamp", "seconds", "sample"],
        default="auto",
        help="Time detection mode. Default: auto.",
    )

    parser.add_argument(
        "--time-col",
        help="Explicit time column name. For your dataset this is probably timestamp.",
    )

    parser.add_argument(
        "--group-cols",
        default="auto",
        help=(
            "Comma-separated grouping columns. Default: auto, which uses "
            "user_id,session_id if present."
        ),
    )

    parser.add_argument(
        "--no-grouping",
        action="store_true",
        help="Disable session/user grouping and treat the file as one recording.",
    )

    parser.add_argument(
        "--valence-out-col",
        default="manual_valence",
        help="Name of output valence label column. Default: manual_valence.",
    )

    parser.add_argument(
        "--arousal-out-col",
        default="manual_arousal",
        help="Name of output arousal label column. Default: manual_arousal.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.eeg_csv):
        raise FileNotFoundError(f"EEG CSV not found: {args.eeg_csv}")

    df = pd.read_csv(args.eeg_csv)

    group_cols = resolve_group_cols(
        df=df,
        group_cols_arg=args.group_cols,
        no_grouping=args.no_grouping,
    )

    if args.time_col:
        if args.time_col not in df.columns:
            raise ValueError(f"--time-col '{args.time_col}' not found in CSV.")

        if args.time_mode == "auto":
            mode, _ = detect_time_column(df)
            time_mode = mode
        else:
            time_mode = args.time_mode

        time_col = args.time_col

    else:
        if args.time_mode == "auto":
            time_mode, time_col = detect_time_column(df)
        else:
            time_mode = args.time_mode

            if time_mode == "timestamp":
                candidates = [
                    "timestamp",
                    "Timestamp",
                    "time_stamp",
                    "TimeStamp",
                    "datetime",
                    "DateTime",
                ]
            elif time_mode == "seconds":
                candidates = [
                    "seconds",
                    "Seconds",
                    "sec",
                    "Sec",
                    "elapsed",
                    "Elapsed",
                    "time",
                    "Time",
                    "time_elapsed",
                    "TimeElapsed",
                ]
            else:
                candidates = [
                    "sample",
                    "Sample",
                    "sample_index",
                    "SampleIndex",
                    "index",
                    "Index",
                ]

            time_col = None
            for c in candidates:
                if c in df.columns:
                    time_col = c
                    break

            if time_col is None:
                raise ValueError(
                    f"Could not find a column for time_mode={time_mode}. "
                    "Provide --time-col explicitly."
                )

    elapsed_seconds = compute_elapsed_seconds(
        df=df,
        mode=time_mode,
        col=time_col,
        group_cols=group_cols,
        sample_rate=args.sample_rate,
    )

    minute_index = build_minute_index(elapsed_seconds)

    out_df = add_time_columns(
        df=df,
        elapsed_seconds=elapsed_seconds,
        minute_index=minute_index,
    )

    if args.generate_template:
        generate_template(
            df=out_df,
            elapsed_seconds=elapsed_seconds,
            group_cols=group_cols,
            out_path=args.generate_template,
            valence_col=args.valence_out_col,
            arousal_col=args.arousal_out_col,
        )

        if not args.annotations and not args.out:
            print("Generated template only. No labeled dataset written.")
            return

    if args.annotations:
        ann = load_annotations(
            annotations_path=args.annotations,
            group_cols=group_cols,
            output_valence_col=args.valence_out_col,
            output_arousal_col=args.arousal_out_col,
        )

        ann = ensure_annotation_group_cols(
            ann=ann,
            df=out_df,
            group_cols=group_cols,
        )

        out_df = merge_annotations(
            df=out_df,
            ann=ann,
            group_cols=group_cols,
            output_valence_col=args.valence_out_col,
            output_arousal_col=args.arousal_out_col,
        )

    out_path = args.out

    if out_path is None:
        stem, ext = os.path.splitext(args.eeg_csv)
        out_path = f"{stem}_with_affect{ext or '.csv'}"

    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Rows written: {len(out_df)}")
    print(f"Time column used: {time_col}")
    print(f"Time mode used: {time_mode}")
    print(f"Group columns used: {group_cols if group_cols else 'none'}")


if __name__ == "__main__":
    main()