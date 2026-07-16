from __future__ import annotations

import os
import time
import csv
import json
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
)

import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices("GPU"))
from ML.model_training import train_lstm, build_eego_lstm_sequences

# IMPORTANT:
# This uses YOUR existing architecture + training loop from ML.model_training
# Assumes you've updated train_lstm to accept X_val/y_val + validation_data
from .ML.model_training import build_eego_lstm_sequences, train_lstm


# ---------------- Data prep ----------------
def load_eego_df(filename: str | Path = "EEGo_labeled.csv") -> pd.DataFrame:
    df = pd.read_csv(filename)
    sort_cols = ["user_id", "session_id"]
    if "time_elapsed" in df.columns:
        sort_cols.append("time_elapsed")
    elif "timestamp" in df.columns:
        sort_cols.append("timestamp")
    return df.sort_values(sort_cols).reset_index(drop=True)


def select_eego_features(df: pd.DataFrame) -> list[str]:
    eeg_prefixes = [
        "F7_",
        "T7_",
        "T8_",
        "F8_",  # Muse alt: ["AF7_", "TP9_", "TP10_", "AF8_"]
    ]
    return [c for c in df.columns if any(c.startswith(p) for p in eeg_prefixes)]


def balance_binary_sequences(X: np.ndarray, y: np.ndarray, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    idx_pos = np.where(y == 1.0)[0]
    idx_neg = np.where(y == 0.0)[0]

    n_pos = len(idx_pos)
    n_neg = len(idx_neg)

    if n_pos == 0 or n_neg == 0 or n_pos == n_neg:
        return X, y

    if n_pos > n_neg:
        keep_pos = rng.choice(idx_pos, size=n_neg, replace=False)
        keep_idx = np.concatenate([keep_pos, idx_neg])
    else:
        keep_neg = rng.choice(idx_neg, size=n_pos, replace=False)
        keep_idx = np.concatenate([keep_neg, idx_pos])

    keep_idx = np.sort(keep_idx)
    return X[keep_idx], y[keep_idx]


# ---------------- Metrics helpers ----------------
def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """
    Returns (metrics_dict, y_pred, confusion_matrix)
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= thresh).astype(int)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    # AUC only defined if both classes are present
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics, y_pred, cm


def summarize_metric_dicts(dicts: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    if not dicts:
        return {}
    keys = list(dicts[0].keys())
    out: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d in dicts], dtype=float)
        out[k] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return out


def print_summary_block(title: str, summary: dict[str, tuple[float, float]]) -> None:
    print(title, flush=True)
    for k, (m, s) in summary.items():
        print(f"    {k}: {m:.4f} (std={s:.4f})", flush=True)


# ---------------- Final training / saving ----------------
def retrain_and_save_best_models(
    features_table: pd.DataFrame,
    feature_cols: list[str],
    *,
    targets: list[str],
    thresh: float,
    fixed_T: int,
    best_params: dict,
    out_dir: str = "models",
) -> dict[str, str]:
    """
    Retrain one final model per target on ALL sessions using best hyperparams,
    then save as .keras. Returns {target: saved_path}.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved_paths: dict[str, str] = {}

    # For final training, use the full dataset as "train".
    full_df = features_table.drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)

    for target_col in targets:
        print("\n==============================", flush=True)
        print(f"FINAL TRAIN | Target: {target_col}", flush=True)
        print("==============================", flush=True)
        print("Full label counts:", flush=True)
        print(full_df[target_col].value_counts(), flush=True)

        X_seq, y_seq = build_eego_lstm_sequences(
            full_df,
            feature_cols=feature_cols,
            target_col=target_col,
            thresh=thresh,
            fixed_T=fixed_T,
        )

        # NOTE: train_lstm uses validation_split internally, so we can pass X_seq as both train/test placeholders.
        model, _, _ = train_lstm(
            X_seq,
            X_seq,
            y_seq,
            y_seq,
            lr=best_params["lr"],
            epochs=best_params["epochs"],
            units=best_params["units"],
            batch_size=best_params["batch_size"],
            patience=best_params["patience"],
            bidirectional=True,
            verbose=2,  # clean epoch-level output for SLURM logs
        )

        save_path = os.path.join(out_dir, f"best_{target_col.lower()}_lstm.keras")
        model.save(save_path)
        saved_paths[target_col] = save_path
        print(f"Saved best final model -> {save_path}", flush=True)

    return saved_paths


# ---------------- Main ----------------
def main() -> None:
    print("Starting EEGo LSTM CV script...", flush=True)

    # Load EEGo data
    df_eego = load_eego_df("dreamer_models/datasets/EEGo_labeled.csv")
    print("EEGo shape:", df_eego.shape, flush=True)

    VALENCE = "affect_valence"
    AROUSAL = "affect_arousal"
    THRESH = 2.5

    session_ids = df_eego["session_id"].unique().tolist()
    print("Number of sessions:", len(session_ids), flush=True)

    counts = df_eego["affect_emotion"].value_counts()
    print(counts, flush=True)
    if all(k in counts for k in ["E", "A", "B", "R"]):
        baseline = (counts["E"] + counts["A"]) / (counts["E"] + counts["A"] + counts["B"] + counts["R"])
        print("Arousal Baseline:", baseline, flush=True)
        
        baseline = (counts["E"] + counts["R"]) / (counts["E"] + counts["A"] + counts["B"] + counts["R"])
        print("Valence Baseline:", baseline, flush=True)

    # Load features table
    features_table = pd.read_csv("dreamer_models/datasets/eego_features.csv")
    feature_cols = select_eego_features(features_table)
    print("n_features:", len(feature_cols), flush=True)
    print(feature_cols, flush=True)

    # ---------------- Hyper-parameter grid ----------------
    param_grid = {
        "lr": [1e-4, 3e-4, 1e-3],
        "epochs": [100, 300, 1000],
        "units": [128, 256, 512, 1024],
        "batch_size": [64, 128, 256],
        "patience": [20],
        "fixed_T": [800, 1500, 2500, 3000],
    }
    # param_grid = {
    #     # Learning rate is usually the #1 driver of stability/speed.
    #     # 1e-4 is safe; 3e-4 and 1e-3 test faster learning.
    #     "lr": [1e-4],
    #     "epochs": [100],
    #     "units": [256, 512],
    #     "batch_size": [64, 256],
    #     "patience": [20],
    #     "fixed_T": [1200, 2500],
    # }


    # ---------------- Leave-k-out settings ----------------
    K_LEAVE_OUT = 5  # informational; folds below are fixed at 4 splits
    rng = np.random.default_rng(42)
    rng.shuffle(session_ids)

    # 4 disjoint test groups (k=5 each when you have 20 sessions)
    folds = np.array_split(session_ids, 4)

    for i, test_sessions in enumerate(folds):
        test_sessions = test_sessions.tolist()
        print(f"Fold {i}: test(k={len(test_sessions)})={sorted(test_sessions)}", flush=True)

    targets = [VALENCE, AROUSAL]

    best_params: dict | None = None
    best_mean_score = -np.inf

    # Track best by your combined metric:
    # Choose which metric to optimize for combined score.
    # Recommended for imbalance: "f1" or "bal_acc". If you want to keep "acc", set to "acc".
    OPT_METRIC = "acc"

    for lr, epochs, units, batch_size, patience, fixed_T in product(
        param_grid["lr"],
        param_grid["epochs"],
        param_grid["units"],
        param_grid["batch_size"],
        param_grid["patience"],
        param_grid["fixed_T"],
    ):
        print("\n#############################################", flush=True)
        print(f"HP COMBO: lr={lr}, epochs={epochs}, units={units}, batch={batch_size}, patience={patience}, fixed_T={fixed_T}", flush=True)
        print("#############################################", flush=True)

        # Store per-fold metrics (dicts) per target
        combo_scores: dict[str, list[dict[str, float]]] = {t: [] for t in targets}

        t0_combo = time.time()

        for fold_i, test_sessions in enumerate(folds):
            print("\n==============================", flush=True)
            print(f"Fold {fold_i} | LEFT OUT SESSIONS: {list(test_sessions)}", flush=True)
            print("==============================", flush=True)

            use_features = features_table.drop(columns=["Unnamed: 0"], errors="ignore")

            mask_test = use_features["session_id"].isin(test_sessions)
            train_df = use_features[~mask_test].reset_index(drop=True)
            user_df = use_features[mask_test].reset_index(drop=True)

            for target_col in targets:
                print(f"\n--- Target: {target_col} ---", flush=True)
                print("Train label counts:", flush=True)
                print(train_df[target_col].value_counts(), flush=True)
                print("Test label counts:", flush=True)
                print(test_df[target_col].value_counts(), flush=True)

                # Build sequences for LSTM
                X_train_seq, y_train_seq = build_eego_lstm_sequences(
                    train_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    thresh=THRESH,
                    fixed_T=fixed_T,
                )
                X_test_seq, y_test_seq = build_eego_lstm_sequences(
                    test_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    thresh=THRESH,
                    fixed_T=fixed_T,
                )

                # Optional balancing (uncomment if desired)
                # X_train_seq, y_train_seq = balance_binary_sequences(X_train_seq, y_train_seq)
                # X_test_seq, y_test_seq = balance_binary_sequences(X_test_seq, y_test_seq)

                # Train
                lstm_model, X_test_eval, y_test_eval = train_lstm(
                    X_train_seq,
                    X_test_seq,
                    y_train_seq,
                    y_test_seq,
                    lr=lr,
                    epochs=epochs,
                    units=units,
                    batch_size=batch_size,
                    patience=patience,
                    bidirectional=True,
                    verbose=2,  # epoch-level logs (good for SLURM)
                )

                # Predict
                y_prob = lstm_model.predict(X_test_eval, batch_size=batch_size, verbose=0).ravel()
                y_true = y_test_eval.astype("int32").ravel()

                fold_metrics, y_pred, cm = compute_binary_metrics(y_true=y_true, y_prob=y_prob, thresh=0.5)

                print(f"\nFold metrics ({target_col}):", flush=True)
                for k, v in fold_metrics.items():
                    print(f"  {k}: {v:.4f}", flush=True)
                print("Confusion matrix:\n", cm, flush=True)
                print("Classification report:", flush=True)
                print(classification_report(y_true, y_pred, digits=4, zero_division=0), flush=True)

                combo_scores[target_col].append(fold_metrics)

        # Summaries
        val_sum = summarize_metric_dicts(combo_scores[VALENCE])
        aro_sum = summarize_metric_dicts(combo_scores[AROUSAL])

        # Combined score (mean of chosen metric across targets)
        mean_val = val_sum.get(OPT_METRIC, (np.nan, np.nan))[0]
        mean_aro = aro_sum.get(OPT_METRIC, (np.nan, np.nan))[0]
        mean_score = float((mean_val + mean_aro) / 2.0)

        dt_combo = time.time() - t0_combo

        print("\n=============================================", flush=True)
        print(f"HP SUMMARY: lr={lr}, epochs={epochs}, units={units}, batch={batch_size}, patience={patience}, fixed_T={fixed_T}", flush=True)
        print(f"Optimizing combined mean of metric: {OPT_METRIC}", flush=True)
        print(f"Combo runtime: {dt_combo/60:.1f} minutes", flush=True)

        print_summary_block("  Valence (mean/std):", val_sum)
        print_summary_block("  Arousal (mean/std):", aro_sum)

        print(f"  Combined mean {OPT_METRIC} score = {mean_score:.4f}", flush=True)
        print("=============================================\n", flush=True)

        if mean_score > best_mean_score:
            best_mean_score = mean_score
            best_params = {
                "lr": lr,
                "epochs": epochs,
                "units": units,
                "batch_size": batch_size,
                "patience": patience,
                "fixed_T": fixed_T,
                "opt_metric": OPT_METRIC,
                "mean_valence_metric": float(mean_val),
                "mean_arousal_metric": float(mean_aro),
                "combined_mean_score": float(mean_score),
                # Keep full summaries too (helps debugging later)
                "valence_summary": {k: {"mean": m, "std": s} for k, (m, s) in val_sum.items()},
                "arousal_summary": {k: {"mean": m, "std": s} for k, (m, s) in aro_sum.items()},
            }
            print(">>> NEW BEST HP FOUND <<<", flush=True)
            print("Best combined score so far:", f"{best_mean_score:.4f}", flush=True)
            print("Best params:", best_params, flush=True)

    # Save best models as .keras (final retrain on all data)
    if best_params is not None:
        saved = retrain_and_save_best_models(
            features_table=features_table,
            feature_cols=feature_cols,
            targets=targets,
            thresh=THRESH,
            fixed_T=int(best_params["fixed_T"]),
            best_params=best_params,
            out_dir="models",
        )
        print("\nSaved model paths:", saved, flush=True)
    else:
        print("No best_params found; skipping final retrain/save.", flush=True)


    print("\n=============================================", flush=True)
    print("Best EEGo leave-k-out combined score:", f"{best_mean_score:.4f}", flush=True)
    print("Best params (with per-target summaries):", best_params, flush=True)
    print("=============================================\n", flush=True)


if __name__ == "__main__":
    # For SLURM: run with `python -u EEGo_models.py`
    main()
