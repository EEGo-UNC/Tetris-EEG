#!/usr/bin/env python3
"""
Add minute-wise valence and arousal annotations to an EEGO CSV.

Features:
- Generate a minute-wise template to annotate valence/arousal after recording
- Merge provided annotations (CSV or JSONL) into EEGO data, adding columns per row
- Robust time detection (timestamp/time/sample with sample-rate)

Usage examples:
1) Generate template to fill:
   python add_minute_affect.py --eego-csv "EEG Prediction/datasets/Bespoke/EEGO.csv" --generate-template "affect_template.csv"

2) Merge annotated template back into EEGO CSV:
   python add_minute_affect.py --eego-csv "EEG Prediction/datasets/Bespoke/EEGO.csv" --annotations "affect_template_filled.csv" --out "EEGO_with_affect.csv"

3) If your file has no time column but has a sample index and you know the sample rate:
   python add_minute_affect.py --eego-csv EEGO.csv --sample-rate 256 --generate-template affect_template.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from typing import Optional, Tuple, List

import pandas as pd


def detect_time_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect time-related columns in the dataframe.
    Returns a tuple of (mode, column_name_or_required):
      - ("timestamp", col) if a timestamp-like column exists
      - ("seconds", col) if a numeric seconds column exists
      - ("sample", col) if a sample index column exists (requires sample_rate)
      - raises ValueError otherwise
    """
    # Candidate timestamp columns (string timestamps)
    timestamp_candidates = ["timestamp", "Timestamp", "time_stamp", "TimeStamp", "date", "datetime", "DateTime"]
    for col in timestamp_candidates:
        if col in df.columns:
            return ("timestamp", col)

    # Candidate seconds columns (float/int seconds)
    seconds_candidates = ["time", "Time", "seconds", "Seconds", "sec", "Sec", "elapsed", "Elapsed"]
    for col in seconds_candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return ("seconds", col)

    # Candidate sample columns (integer index)
    sample_candidates = ["sample", "Sample", "index", "Index"]
    for col in sample_candidates:
        if col in df.columns and pd.api.types.is_integer_dtype(df[col]):
            return ("sample", col)

    raise ValueError(
        "Could not detect a time-related column. Provide --sample-rate if you have a 'sample' column, "
        "or ensure a 'timestamp' or 'time' (seconds) column exists."
    )


def compute_elapsed_seconds(
    df: pd.DataFrame,
    mode: str,
    col: str,
    sample_rate: Optional[float] = None
) -> pd.Series:
    """
    Compute elapsed seconds for each row based on the detected time mode/column.
    """
    if mode == "timestamp":
        # Attempt to parse datetimes; compute elapsed from the first timestamp
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.isna().all():
            raise ValueError(f"Column '{col}' could not be parsed as datetimes.")
        first_valid = parsed.dropna().iloc[0]
        elapsed = (parsed - first_valid).dt.total_seconds()
        return elapsed

    if mode == "seconds":
        # Assume column already represents seconds; normalize to start from 0
        s = df[col].astype(float)
        return s - float(s.iloc[0])

    if mode == "sample":
        if sample_rate is None or sample_rate <= 0:
            raise ValueError("sample_rate must be provided and > 0 when using a sample index column.")
        s = df[col].astype(float)
        return (s - float(s.iloc[0])) / float(sample_rate)

    raise ValueError(f"Unsupported time mode: {mode}")


def build_minute_index(elapsed_seconds: pd.Series) -> pd.Series:
    """
    Compute minute_index = floor(elapsed_seconds / 60).
    """
    return (elapsed_seconds // 60).astype(int)


def compute_minute_boundaries(elapsed_seconds: pd.Series) -> List[Tuple[int, float, float]]:
    """
    Return a list of (minute, start_sec, end_sec) across the recording duration.
    """
    total_duration = float(elapsed_seconds.max())
    last_minute = int(math.floor(total_duration / 60.0))
    boundaries = []
    for m in range(0, last_minute + 1):
        start = m * 60.0
        end = min((m + 1) * 60.0, total_duration)
        boundaries.append((m, start, end))
    return boundaries


def load_annotations(annotations_path: str) -> pd.DataFrame:
    """
    Load annotations from CSV or JSONL into a DataFrame with columns:
      minute, valence, arousal
    CSV columns accepted:
      - minute,valence,arousal
      - or start_minute,end_minute,valence,arousal (expanded per minute)
    JSONL entries accepted:
      - {"minute": 0, "valence": 5, "arousal": 4}
      - {"start_minute": 0, "end_minute": 3, "valence": 5, "arousal": 4}
    """
    ext = os.path.splitext(annotations_path)[1].lower()
    records: List[dict] = []

    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(annotations_path, sep=sep)
        cols = [c.lower() for c in df.columns]
        if {"minute", "valence", "arousal"}.issubset(cols):
            # Normalize column names
            df = df.rename(columns={c: c.lower() for c in df.columns})
            for _, row in df.iterrows():
                records.append({
                    "minute": int(row["minute"]),
                    "valence": row["valence"],
                    "arousal": row["arousal"],
                })
        elif {"start_minute", "end_minute", "valence", "arousal"}.issubset(cols):
            df = df.rename(columns={c: c.lower() for c in df.columns})
            for _, row in df.iterrows():
                start_m = int(row["start_minute"])
                end_m = int(row["end_minute"])
                for m in range(start_m, end_m + 1):
                    records.append({
                        "minute": m,
                        "valence": row["valence"],
                        "arousal": row["arousal"],
                    })
        else:
            raise ValueError("Annotations CSV must have columns minute,valence,arousal OR start_minute,end_minute,valence,arousal")

    elif ext in [".jsonl", ".json"]:
        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj = {k.lower(): v for k, v in obj.items()}
                if "minute" in obj:
                    records.append({
                        "minute": int(obj["minute"]),
                        "valence": obj.get("valence"),
                        "arousal": obj.get("arousal"),
                    })
                elif "start_minute" in obj and "end_minute" in obj:
                    start_m = int(obj["start_minute"])
                    end_m = int(obj["end_minute"])
                    for m in range(start_m, end_m + 1):
                        records.append({
                            "minute": m,
                            "valence": obj.get("valence"),
                            "arousal": obj.get("arousal"),
                        })
                else:
                    raise ValueError("Annotations JSONL entries must include 'minute' or 'start_minute' and 'end_minute'.")
    else:
        raise ValueError("Unsupported annotations format. Use .csv, .tsv, .jsonl, or .json")

    ann_df = pd.DataFrame.from_records(records)
    if ann_df.empty:
        raise ValueError("No valid annotations found in file.")
    # Deduplicate by minute keeping last provided entry
    ann_df = ann_df.drop_duplicates(subset=["minute"], keep="last").sort_values("minute")
    return ann_df


def generate_template(
    df: pd.DataFrame,
    elapsed_seconds: pd.Series,
    out_path: str
) -> None:
    """
    Generate a CSV template with minute,start_sec,end_sec,valence,arousal
    """
    boundaries = compute_minute_boundaries(elapsed_seconds)
    tmpl = pd.DataFrame(
        [{"minute": m, "start_sec": s, "end_sec": e, "valence": "", "arousal": ""} for (m, s, e) in boundaries]
    )
    tmpl.to_csv(out_path, index=False)
    print(f"Template written: {out_path} (minutes: {len(tmpl)})")


def merge_annotations(
    df: pd.DataFrame,
    minute_index: pd.Series,
    ann_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge annotations with EEGO data by minute, returning augmented DataFrame with
    columns: affect_minute, valence, arousal
    """
    # Left-join by minute
    join_df = pd.DataFrame({"__minute": minute_index})
    ann_df2 = ann_df.rename(columns={"minute": "__minute"})
    merged = pd.concat([df.reset_index(drop=True), join_df], axis=1).merge(
        ann_df2, how="left", on="__minute"
    )
    # Rename columns
    merged = merged.rename(columns={"__minute": "affect_minute"})
    return merged


def main():
    parser = argparse.ArgumentParser(description="Add minute-wise valence/arousal to EEGO CSV.")
    parser.add_argument("--eego-csv", required=True, help="Path to EEGO CSV file")
    parser.add_argument("--out", help="Output CSV path (default: <input>_with_affect.csv)")
    parser.add_argument("--generate-template", help="Path to write a minute-wise annotation template CSV")
    parser.add_argument("--annotations", help="Path to annotations file (CSV/TSV/JSONL/JSON)")
    parser.add_argument("--sample-rate", type=float, help="Sample rate (Hz) if using sample index")
    parser.add_argument("--time-mode", choices=["auto", "timestamp", "seconds", "sample"], default="auto",
                        help="Time detection mode (auto by default)")
    parser.add_argument("--time-col", help="Explicit time column name (optional)")

    args = parser.parse_args()

    if not os.path.exists(args.eego_csv):
        raise FileNotFoundError(f"EEGO CSV not found: {args.eego_csv}")

    df = pd.read_csv(args.eego_csv)

    # Determine time mode/column
    if args.time_mode == "auto":
        mode, detected_col = detect_time_columns(df)
    else:
        if args.time_col is None:
            # Try to auto-pick based on mode
            mode, detected_col = detect_time_columns(df)
            if mode != args.time_mode:
                # If mismatch, and user provided explicit mode, enforce
                # But ensure the required column exists
                if args.time_mode == "timestamp":
                    candidates = [c for c in df.columns if c.lower() in {"timestamp", "datetime", "date", "time"}]
                    if not candidates:
                        raise ValueError("Explicit time-mode=timestamp but no timestamp-like column found; provide --time-col.")
                    detected_col = candidates[0]
                elif args.time_mode == "seconds":
                    candidates = [c for c in df.columns if c.lower() in {"time", "seconds", "sec", "elapsed"}]
                    if not candidates:
                        raise ValueError("Explicit time-mode=seconds but no seconds-like column found; provide --time-col.")
                    detected_col = candidates[0]
                elif args.time_mode == "sample":
                    candidates = [c for c in df.columns if c.lower() in {"sample", "index"}]
                    if not candidates:
                        raise ValueError("Explicit time-mode=sample but no sample/index column found; provide --time-col.")
                    detected_col = candidates[0]
                mode = args.time_mode
        else:
            mode = args.time_mode
            detected_col = args.time_col
            if detected_col not in df.columns:
                raise ValueError(f"--time-col '{detected_col}' not found in CSV.")

    elapsed = compute_elapsed_seconds(df, mode, detected_col, sample_rate=args.sample_rate)
    minute_idx = build_minute_index(elapsed)

    # Generate template
    if args.generate_template:
        generate_template(df, elapsed, args.generate_template)
        # If only template requested and no merge, we can exit
        if not args.annotations and not args.out:
            return

    # Merge annotations
    if args.annotations:
        ann_df = load_annotations(args.annotations)
        out_df = merge_annotations(df, minute_idx, ann_df)
    else:
        out_df = df.copy()
        out_df.insert(len(out_df.columns), "affect_minute", minute_idx.values)

    out_path = args.out or os.path.splitext(args.eego_csv)[0] + "_with_affect.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


