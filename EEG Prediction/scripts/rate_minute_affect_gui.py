#!/usr/bin/env python3
"""
Interactive GUI to annotate minute-wise valence and arousal for an EEGO CSV.

Workflow:
- You pass your EEGO CSV file.
- The tool computes how long the recording is (via timestamp or time_elapsed).
- It then shows a simple GUI, one minute at a time:
    Minute 0 (0.0s – 59.9s)
    [Valence slider]  [Arousal slider]
    [Prev] [Next] [Save & Exit]
- You choose ratings on configurable scales (e.g. 1–9 or 1–5).
- On Save, it writes:
    1) A minute-wise annotations CSV (minute,start_sec,end_sec,valence,arousal)
    2) An updated EEGO CSV with those ratings attached to every row.

Example:
    python rate_minute_affect_gui.py \\
        --eego-csv "EEG Prediction/datasets/Bespoke/EEGO.csv" \\
        --scale-min 1 --scale-max 9
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Optional, List, Tuple

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox


def detect_time_column(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Detect a time-like column and its mode.
    Returns (mode, column_name) where mode is one of: 'timestamp', 'seconds'.

    For your EEGO.csv, this will likely pick 'time_elapsed' (seconds)
    or 'timestamp' (ISO datetime).
    """
    # Use numeric time_elapsed if available
    if "time_elapsed" in df.columns and pd.api.types.is_numeric_dtype(df["time_elapsed"]):
        return "seconds", "time_elapsed"

    # Fallbacks, similar to the other script
    timestamp_candidates = ["timestamp", "Timestamp", "datetime", "DateTime"]
    for col in timestamp_candidates:
        if col in df.columns:
            return "timestamp", col

    seconds_candidates = ["time", "Time", "seconds", "Seconds", "sec", "Sec", "elapsed", "Elapsed"]
    for col in seconds_candidates:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return "seconds", col

    raise ValueError(
        "Could not detect a time column. "
        "Please add a numeric 'time_elapsed' column or a 'timestamp' column."
    )


def compute_elapsed_seconds(df: pd.DataFrame, mode: str, col: str) -> pd.Series:
    """Compute elapsed seconds for each row based on the chosen column."""
    if mode == "seconds":
        s = df[col].astype(float)
        return s - float(s.iloc[0])

    if mode == "timestamp":
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.isna().all():
            raise ValueError(f"Column '{col}' could not be parsed as datetimes.")
        first_valid = parsed.dropna().iloc[0]
        elapsed = (parsed - first_valid).dt.total_seconds()
        return elapsed

    raise ValueError(f"Unsupported time mode: {mode}")


def compute_minute_boundaries(elapsed_seconds: pd.Series) -> List[Tuple[int, float, float]]:
    """Return a list of (minute, start_sec, end_sec) across the recording."""
    total_duration = float(elapsed_seconds.max())
    last_minute = int(math.floor(total_duration / 60.0))
    boundaries = []
    for m in range(0, last_minute + 1):
        start = m * 60.0
        end = min((m + 1) * 60.0, total_duration)
        boundaries.append((m, start, end))
    return boundaries


class MinuteRatingGUI:
    def __init__(
        self,
        minute_bounds: List[Tuple[int, float, float]],
        scale_min: int,
        scale_max: int,
    ) -> None:
        self.minute_bounds = minute_bounds
        self.scale_min = scale_min
        self.scale_max = scale_max

        # Ratings storage: list of dicts {'minute': int, 'valence': Optional[int], 'arousal': Optional[int]}
        self.ratings = [
            {"minute": m, "valence": None, "arousal": None}
            for (m, _, _) in minute_bounds
        ]
        self.current_idx = 0

        self.root = tk.Tk()
        self.root.title("Minute-wise Valence/Arousal Rating")

        self._build_ui()
        self._update_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Label for minute info
        self.minute_label = ttk.Label(frame, text="", font=("TkDefaultFont", 12, "bold"))
        self.minute_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # Valence
        ttk.Label(frame, text="Valence:").grid(row=1, column=0, sticky="w")
        self.valence_var = tk.IntVar()
        self.valence_scale = ttk.Scale(
            frame,
            from_=self.scale_min,
            to=self.scale_max,
            orient="horizontal",
            command=self._on_valence_scale,
        )
        self.valence_scale.grid(row=1, column=1, sticky="ew", padx=(5, 0))
        self.valence_value_label = ttk.Label(frame, text="")
        self.valence_value_label.grid(row=1, column=2, sticky="w", padx=(5, 0))

        # Arousal
        ttk.Label(frame, text="Arousal:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.arousal_var = tk.IntVar()
        self.arousal_scale = ttk.Scale(
            frame,
            from_=self.scale_min,
            to=self.scale_max,
            orient="horizontal",
            command=self._on_arousal_scale,
        )
        self.arousal_scale.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        self.arousal_value_label = ttk.Label(frame, text="")
        self.arousal_value_label.grid(row=2, column=2, sticky="w", padx=(5, 0), pady=(5, 0))

        frame.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        btn_frame.columnconfigure((0, 1, 2), weight=1)

        self.prev_button = ttk.Button(btn_frame, text="← Previous", command=self._prev_minute)
        self.prev_button.grid(row=0, column=0, sticky="w")

        self.next_button = ttk.Button(btn_frame, text="Next →", command=self._next_minute)
        self.next_button.grid(row=0, column=1)

        self.save_button = ttk.Button(btn_frame, text="Save & Exit", command=self._on_save)
        self.save_button.grid(row=0, column=2, sticky="e")

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._prev_minute())
        self.root.bind("<Right>", lambda e: self._next_minute())
        self.root.bind("<Return>", lambda e: self._next_minute())

    def _on_valence_scale(self, value: str) -> None:
        v = int(round(float(value)))
        self.valence_var.set(v)
        self.valence_value_label.config(text=str(v))
        self.ratings[self.current_idx]["valence"] = v

    def _on_arousal_scale(self, value: str) -> None:
        v = int(round(float(value)))
        self.arousal_var.set(v)
        self.arousal_value_label.config(text=str(v))
        self.ratings[self.current_idx]["arousal"] = v

    def _update_ui(self) -> None:
        m, start, end = self.minute_bounds[self.current_idx]
        self.minute_label.config(
            text=f"Minute {m}  ({start:.1f}s – {end:.1f}s)  "
                 f"[{self.current_idx + 1} / {len(self.minute_bounds)}]"
        )

        # Restore previous ratings or set to mid-scale
        rating = self.ratings[self.current_idx]
        default_val = (self.scale_min + self.scale_max) // 2

        val = rating["valence"] if rating["valence"] is not None else default_val
        aro = rating["arousal"] if rating["arousal"] is not None else default_val

        self.valence_scale.set(val)
        self.arousal_scale.set(aro)
        self.valence_value_label.config(text=str(val))
        self.arousal_value_label.config(text=str(aro))

        self.valence_var.set(val)
        self.arousal_var.set(aro)

        # Enable/disable navigation buttons
        self.prev_button.config(state=("disabled" if self.current_idx == 0 else "normal"))
        self.next_button.config(state=("disabled" if self.current_idx == len(self.minute_bounds) - 1 else "normal"))

    def _prev_minute(self) -> None:
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_ui()

    def _next_minute(self) -> None:
        if self.current_idx < len(self.minute_bounds) - 1:
            self.current_idx += 1
            self._update_ui()

    def _on_save(self) -> None:
        # Check if any minutes are still None – optional, we can allow partial
        missing = [
            r["minute"] for r in self.ratings
            if r["valence"] is None or r["arousal"] is None
        ]
        if missing:
            if not messagebox.askyesno(
                "Incomplete ratings",
                f"You have not rated minutes: {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
                f"Save anyway?"
            ):
                return
        self.root.quit()

    def run(self) -> List[dict]:
        self.root.mainloop()
        self.root.destroy()
        return self.ratings


def main():
    parser = argparse.ArgumentParser(description="GUI tool to rate minute-wise valence/arousal for EEGO data.")
    parser.add_argument("--eego-csv", required=True, help="Path to EEGO CSV (e.g., EEGO.csv)")
    parser.add_argument("--scale-min", type=int, default=1, help="Minimum rating value (default: 1)")
    parser.add_argument("--scale-max", type=int, default=9, help="Maximum rating value (default: 9)")
    parser.add_argument("--time-col", help="Optional explicit time column name (default: auto-detect)")
    parser.add_argument("--time-mode", choices=["auto", "seconds", "timestamp"], default="auto",
                        help="Time interpretation (default: auto)")
    parser.add_argument("--out-annotations", help="Where to save minute-wise annotations CSV "
                                                  "(default: <EEGO>_minute_affect.csv)")
    parser.add_argument("--out-eego", help="Where to save EEGO with attached minute-wise affect "
                                           "(default: <EEGO>_with_minute_affect.csv)")

    args = parser.parse_args()

    if args.scale_min >= args.scale_max:
        raise ValueError("scale-min must be < scale-max.")

    if not os.path.exists(args.eego_csv):
        raise FileNotFoundError(f"EEGO CSV not found: {args.eego_csv}")

    df = pd.read_csv(args.eego_csv)

    # Time detection
    if args.time_mode == "auto":
        mode, col = detect_time_column(df)
    else:
        if args.time_col is None:
            # Let auto detect, but enforce type (seconds vs timestamp)
            mode, col = detect_time_column(df)
            if args.time_mode != "auto" and args.time_mode != mode:
                # Overrule if user insisted, but they must provide a valid column
                if args.time_mode == "seconds":
                    if "time_elapsed" in df.columns:
                        col = "time_elapsed"
                    else:
                        raise ValueError("time-mode=seconds but no numeric 'time_elapsed' column; use --time-col.")
                elif args.time_mode == "timestamp":
                    if "timestamp" in df.columns:
                        col = "timestamp"
                    else:
                        raise ValueError("time-mode=timestamp but no 'timestamp' column; use --time-col.")
                mode = args.time_mode
        else:
            col = args.time_col
            if col not in df.columns:
                raise ValueError(f"time column '{col}' not found in CSV.")
            if args.time_mode == "auto":
                # Guess based on dtype
                if pd.api.types.is_numeric_dtype(df[col]):
                    mode = "seconds"
                else:
                    mode = "timestamp"
            else:
                mode = args.time_mode

    elapsed = compute_elapsed_seconds(df, mode, col)
    minute_bounds = compute_minute_boundaries(elapsed)

    if not minute_bounds:
        raise ValueError("Recording too short to compute minutes.")

    # Run GUI
    gui = MinuteRatingGUI(minute_bounds, scale_min=args.scale_min, scale_max=args.scale_max)
    ratings = gui.run()

    # Build annotations DataFrame
    records = []
    for (m, start, end), r in zip(minute_bounds, ratings):
        records.append({
            "minute": m,
            "start_sec": start,
            "end_sec": end,
            "valence": r["valence"],
            "arousal": r["arousal"],
        })
    ann_df = pd.DataFrame.from_records(records)

    # Save annotations CSV
    base, ext = os.path.splitext(args.eego_csv)
    ann_out = args.out_annotations or f"{base}_minute_affect.csv"
    ann_df.to_csv(ann_out, index=False)

    # Attach ratings to EEGO by minute index
    minute_idx = (elapsed // 60).astype(int)
    join_df = pd.DataFrame({"__minute": minute_idx})
    ann_df2 = ann_df.rename(columns={"minute": "__minute"})
    merged = pd.concat([df.reset_index(drop=True), join_df], axis=1).merge(
        ann_df2, how="left", on="__minute"
    )
    merged = merged.rename(columns={"__minute": "affect_minute"})

    eego_out = args.out_eego or f"{base}_with_minute_affect.csv"
    merged.to_csv(eego_out, index=False)

    print(f"Saved minute-wise annotations: {ann_out}")
    print(f"Saved EEGO with attached affect: {eego_out}")


if __name__ == "__main__":
    main()


