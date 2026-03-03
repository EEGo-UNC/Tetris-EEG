from __future__ import annotations

import os
import time
from itertools import product

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
)

import tensorflow as tf

# IMPORTANT:
# This uses YOUR existing architecture + training loop from ML.model_training
# Assumes you've updated train_lstm to accept X_val/y_val + validation_data
from ML.model_training import build_eego_lstm_sequences, train_lstm


# ============================================================
#                     Data prep helpers
# ============================================================
def load_eego_df(filename: str = "dreamer_models/datasets/EEGo_labeled.csv") -> pd.DataFrame:
    df = pd.read_csv(filename)
    sort_cols = ["user_id", "session_id"]
    if "time_elapsed" in df.columns:
        sort_cols.append("time_elapsed")
    elif "timestamp" in df.columns:
        sort_cols.append("timestamp")
    return df.sort_values(sort_cols).reset_index(drop=True)


def select_eego_features(df: pd.DataFrame) -> list[str]:
    eeg_prefixes = [
        "F7_", "T7_", "T8_", "F8_",  # Muse alt: ["AF7_", "TP9_", "TP10_", "AF8_"]
    ]
    return [c for c in df.columns if any(c.startswith(p) for p in eeg_prefixes)]


def split_sequences_val_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_frac: float = 0.30,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a single user's sequences into (val, test).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = int(X.shape[0])
    if n == 0:
        return X[:0], y[:0], X[:0], y[:0]
    if n == 1:
        # can't split; we refuse to "test" if we'd be validating on the same single seq
        return X[:0], y[:0], X[:0], y[:0]

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_val = max(1, int(round(val_frac * n)))
    if n_val >= n:
        n_val = n - 1

    val_idx = idx[:n_val]
    test_idx = idx[n_val:]

    return X[val_idx], y[val_idx], X[test_idx], y[test_idx]


# ============================================================
#                    Metrics helpers
# ============================================================
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics, cm


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


# ============================================================
#                 Final training / saving best
# ============================================================
def retrain_and_save_best_models(
    features_table: pd.DataFrame,
    feature_cols: list[str],
    *,
    valence_col: str,
    arousal_col: str,
    thresh: float,
    fixed_T: int,
    best_params: dict,
    out_dir: str = "models",
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    full_df = features_table.drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)

    # VALENCE
    print("\n==============================", flush=True)
    print("FINAL TRAIN | Target: VALENCE (binary)", flush=True)
    print("==============================", flush=True)
    X_v, y_v = build_eego_lstm_sequences(
        full_df,
        feature_cols=feature_cols,
        target_col=valence_col,
        thresh=thresh,
        fixed_T=fixed_T,
    )
    print("Sequence-level valence_bin counts:", np.bincount(y_v.astype(int), minlength=2), flush=True)

    model_v, _, _ = train_lstm(
        X_v,
        X_v,
        y_v,
        y_v,
        units=best_params["units"],
        dropout=best_params["dropout"],
        recurrent_dropout=best_params["recurrent_dropout"],
        lr=best_params["lr"],
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        bidirectional=best_params["bidirectional"],
        patience=best_params["patience"],
        verbose=best_params["verbose"],
        random_seed=best_params["random_seed"],
    )
    val_path = os.path.join(out_dir, "best_valence_4_sensor.keras")
    model_v.save(val_path)
    print(f"Saved -> {val_path}", flush=True)

    # AROUSAL
    print("\n==============================", flush=True)
    print("FINAL TRAIN | Target: AROUSAL (binary)", flush=True)
    print("==============================", flush=True)
    X_a, y_a = build_eego_lstm_sequences(
        full_df,
        feature_cols=feature_cols,
        target_col=arousal_col,
        thresh=thresh,
        fixed_T=fixed_T,
    )
    print("Sequence-level arousal_bin counts:", np.bincount(y_a.astype(int), minlength=2), flush=True)

    model_a, _, _ = train_lstm(
        X_a,
        X_a,
        y_a,
        y_a,
        units=best_params["units"],
        dropout=best_params["dropout"],
        recurrent_dropout=best_params["recurrent_dropout"],
        lr=best_params["lr"],
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        bidirectional=best_params["bidirectional"],
        patience=best_params["patience"],
        verbose=best_params["verbose"],
        random_seed=best_params["random_seed"],
    )
    aro_path = os.path.join(out_dir, "best_arousal_4_sensor.keras")
    model_a.save(aro_path)
    print(f"Saved -> {aro_path}", flush=True)

    return val_path, aro_path


# ============================================================
#                         Main
# ============================================================
def main() -> None:
    print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    print("Starting EEGo LOSOCV (left-out user used for validation)...", flush=True)

    # Convenience / sanity checks
    df_eego = load_eego_df("dreamer_models/datasets/EEGo_labeled.csv")
    print("EEGo shape:", df_eego.shape, flush=True)

    VALENCE = "affect_valence"
    AROUSAL = "affect_arousal"
    THRESH = 2.5

    # Features table (must include EEG features + labels + grouping cols)
    features_table = pd.read_csv("dreamer_models/datasets/eego_features.csv")
    feature_cols = select_eego_features(features_table)
    print("n_features:", len(feature_cols), flush=True)

    # LOSOCV users (leave one user out)
    user_ids = sorted(features_table["user_id"].dropna().unique().tolist())
    print("Number of users (LOSOCV folds):", len(user_ids), flush=True)
    print("Users:", user_ids, flush=True)

    # NOTE: build_eego_lstm_sequences expects fixed_T not None
    FIXED_T = 2500

    LEFT_OUT_VAL_FRAC = 0.50

    # ---------------- Hyper-parameter grid ----------------
    param_grid = {
        "lr": [1e-4],
        "epochs": [100],
        "units": [512],
        "batch_size": [256],
        "patience": [10],
        "dropout": [0.20],
        "recurrent_dropout": [0.1],
        "bidirectional": [True],
        "random_seed": [42],
        "verbose": [2],
    }

    best_params: dict | None = None
    best_mean_score = -np.inf
    OPT_METRIC = "bal_acc"

    for lr, epochs, units, batch_size, patience, dropout, recurrent_dropout, bidirectional, random_seed, verbose in product(
        param_grid["lr"],
        param_grid["epochs"],
        param_grid["units"],
        param_grid["batch_size"],
        param_grid["patience"],
        param_grid["dropout"],
        param_grid["recurrent_dropout"],
        param_grid["bidirectional"],
        param_grid["random_seed"],
        param_grid["verbose"],
    ):
        print("\n#############################################", flush=True)
        print(
            f"HP COMBO: lr={lr}, epochs={epochs}, units={units}, batch={batch_size}, "
            f"patience={patience}, dropout={dropout}, rec_drop={recurrent_dropout}, "
            f"bidir={bidirectional}, seed={random_seed}, fixed_T={FIXED_T}, verbose={verbose}, "
            f"left_out_val_frac={LEFT_OUT_VAL_FRAC}",
            flush=True,
        )
        print("#############################################", flush=True)

        val_scores: list[dict[str, float]] = []
        aro_scores: list[dict[str, float]] = []
        t0_combo = time.time()

        # ===================== LOSOCV LOOP =====================
        for fold_i, test_user in enumerate(user_ids):
            print("\n==============================", flush=True)
            print(f"Fold {fold_i+1}/{len(user_ids)} | LEFT OUT USER: {test_user}", flush=True)
            print("==============================", flush=True)

            use_features = features_table.drop(columns=["Unnamed: 0"], errors="ignore")

            mask_test = use_features["user_id"] == test_user
            train_df = use_features[~mask_test].reset_index(drop=True)
            user_df = use_features[mask_test].reset_index(drop=True)

            if len(user_df) == 0:
                print(f"WARNING: user_id={test_user} has 0 rows in features_table. Skipping fold.", flush=True)
                continue

            # ---------------- VALENCE ----------------
            X_train_v, y_train_v = build_eego_lstm_sequences(
                train_df,
                feature_cols=feature_cols,
                target_col=VALENCE,
                thresh=THRESH,
                fixed_T=FIXED_T,
            )

            X_user_v, y_user_v = build_eego_lstm_sequences(
                user_df,
                feature_cols=feature_cols,
                target_col=VALENCE,
                thresh=THRESH,
                fixed_T=FIXED_T,
            )

            X_val_v, y_val_v, X_test_v, y_test_v = split_sequences_val_test(
                X_user_v,
                y_user_v,
                val_frac=LEFT_OUT_VAL_FRAC,
                seed=10_000 + fold_i,
            )

            if X_test_v.shape[0] == 0 or X_val_v.shape[0] == 0:
                print(
                    f"WARNING: user_id={test_user} has insufficient sequences for VALENCE "
                    f"(val={X_val_v.shape[0]}, test={X_test_v.shape[0]}). Skipping.",
                    flush=True,
                )
            else:
                print("Train val_bin counts:", np.bincount(y_train_v.astype(int), minlength=2), flush=True)
                print("VAL   val_bin counts:", np.bincount(y_val_v.astype(int), minlength=2), flush=True)
                print("TEST  val_bin counts:", np.bincount(y_test_v.astype(int), minlength=2), flush=True)

                model_v, X_v_eval, y_v_eval = train_lstm(
                    X_train_v,
                    X_test_v,
                    y_train_v,
                    y_test_v,
                    units=units,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    bidirectional=bidirectional,
                    patience=patience,
                    verbose=verbose,
                    random_seed=random_seed,
                    X_val=X_val_v,
                    y_val=y_val_v,
                )

                p_v = model_v.predict(X_v_eval, batch_size=batch_size, verbose=0).reshape(-1)
                y_v_pred = (p_v >= 0.5).astype(np.int32)
                y_v_true = y_v_eval.astype(np.int32)

                fold_metrics_v, cm_v = compute_binary_metrics(y_true=y_v_true, y_pred=y_v_pred)
                print("\nFold metrics (VALENCE | TEST split):", flush=True)
                for k, v in fold_metrics_v.items():
                    print(f"  {k}: {v:.4f}", flush=True)
                print("Confusion matrix (rows=true, cols=pred):\n", cm_v, flush=True)
                print(
                    classification_report(
                        y_v_true,
                        y_v_pred,
                        labels=[0, 1],
                        target_names=["low", "high"],
                        digits=4,
                        zero_division=0,
                    ),
                    flush=True,
                )
                val_scores.append(fold_metrics_v)

            # ---------------- AROUSAL ----------------
            X_train_a, y_train_a = build_eego_lstm_sequences(
                train_df,
                feature_cols=feature_cols,
                target_col=AROUSAL,
                thresh=THRESH,
                fixed_T=FIXED_T,
            )

            X_user_a, y_user_a = build_eego_lstm_sequences(
                user_df,
                feature_cols=feature_cols,
                target_col=AROUSAL,
                thresh=THRESH,
                fixed_T=FIXED_T,
            )

            X_val_a, y_val_a, X_test_a, y_test_a = split_sequences_val_test(
                X_user_a,
                y_user_a,
                val_frac=LEFT_OUT_VAL_FRAC,
                seed=20_000 + fold_i,
            )

            if X_test_a.shape[0] == 0 or X_val_a.shape[0] == 0:
                print(
                    f"WARNING: user_id={test_user} has insufficient sequences for AROUSAL "
                    f"(val={X_val_a.shape[0]}, test={X_test_a.shape[0]}). Skipping.",
                    flush=True,
                )
            else:
                print("Train aro_bin counts:", np.bincount(y_train_a.astype(int), minlength=2), flush=True)
                print("VAL   aro_bin counts:", np.bincount(y_val_a.astype(int), minlength=2), flush=True)
                print("TEST  aro_bin counts:", np.bincount(y_test_a.astype(int), minlength=2), flush=True)

                model_a, X_a_eval, y_a_eval = train_lstm(
                    X_train_a,
                    X_test_a,
                    y_train_a,
                    y_test_a,
                    units=units,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    bidirectional=bidirectional,
                    patience=patience,
                    verbose=verbose,
                    random_seed=random_seed,
                    X_val=X_val_a,
                    y_val=y_val_a,
                )

                p_a = model_a.predict(X_a_eval, batch_size=batch_size, verbose=0).reshape(-1)
                y_a_pred = (p_a >= 0.5).astype(np.int32)
                y_a_true = y_a_eval.astype(np.int32)

                fold_metrics_a, cm_a = compute_binary_metrics(y_true=y_a_true, y_pred=y_a_pred)
                print("\nFold metrics (AROUSAL | TEST split):", flush=True)
                for k, v in fold_metrics_a.items():
                    print(f"  {k}: {v:.4f}", flush=True)
                print("Confusion matrix (rows=true, cols=pred):\n", cm_a, flush=True)
                print(
                    classification_report(
                        y_a_true,
                        y_a_pred,
                        labels=[0, 1],
                        target_names=["low", "high"],
                        digits=4,
                        zero_division=0,
                    ),
                    flush=True,
                )
                aro_scores.append(fold_metrics_a)
        # =================== END LOSOCV LOOP ===================

        val_summary = summarize_metric_dicts(val_scores)
        aro_summary = summarize_metric_dicts(aro_scores)
        val_mean = val_summary.get(OPT_METRIC, (np.nan, np.nan))[0]
        aro_mean = aro_summary.get(OPT_METRIC, (np.nan, np.nan))[0]
        combined_mean = float(np.nanmean([val_mean, aro_mean]))

        dt_combo = time.time() - t0_combo

        print("\n=============================================", flush=True)
        print(
            f"HP SUMMARY: lr={lr}, epochs={epochs}, units={units}, batch={batch_size}, "
            f"patience={patience}, dropout={dropout}, rec_drop={recurrent_dropout}, "
            f"bidir={bidirectional}, seed={random_seed}, fixed_T={FIXED_T}, verbose={verbose}, "
            f"left_out_val_frac={LEFT_OUT_VAL_FRAC}",
            flush=True,
        )
        print(f"Optimizing metric: mean({OPT_METRIC}_val, {OPT_METRIC}_aro)", flush=True)
        print(f"Combo runtime: {dt_combo/60:.1f} minutes", flush=True)
        print_summary_block("  VALENCE (mean/std):", val_summary)
        print_summary_block("  AROUSAL (mean/std):", aro_summary)
        print(f"  Combined mean score = {combined_mean:.4f}", flush=True)
        print("=============================================\n", flush=True)

        if combined_mean > best_mean_score:
            best_mean_score = float(combined_mean)
            best_params = {
                "lr": lr,
                "epochs": epochs,
                "units": units,
                "batch_size": batch_size,
                "patience": patience,
                "dropout": dropout,
                "recurrent_dropout": recurrent_dropout,
                "bidirectional": bidirectional,
                "random_seed": random_seed,
                "verbose": verbose,
                "fixed_T": FIXED_T,
                "thresh": THRESH,
                "left_out_val_frac": LEFT_OUT_VAL_FRAC,
                "opt_metric": OPT_METRIC,
                "combined_mean_score": combined_mean,
                "val_summary": {k: {"mean": m, "std": s} for k, (m, s) in val_summary.items()},
                "aro_summary": {k: {"mean": m, "std": s} for k, (m, s) in aro_summary.items()},
            }
            print(">>> NEW BEST HP FOUND <<<", flush=True)
            print("Best score so far:", f"{best_mean_score:.4f}", flush=True)
            print("Best params:", best_params, flush=True)

    # Final retrain/save (two separate models)
    if best_params is not None:
        val_path, aro_path = retrain_and_save_best_models(
            features_table=features_table,
            feature_cols=feature_cols,
            valence_col=VALENCE,
            arousal_col=AROUSAL,
            thresh=float(best_params["thresh"]),
            fixed_T=int(best_params["fixed_T"]),
            best_params=best_params,
            out_dir="models",
        )
        print("\nSaved final model paths:", flush=True)
        print("  Valence:", val_path, flush=True)
        print("  Arousal:", aro_path, flush=True)
    else:
        print("No best_params found; skipping final retrain/save.", flush=True)

    print("\n=============================================", flush=True)
    print("Best combined score:", f"{best_mean_score:.4f}", flush=True)
    print("Best params:", best_params, flush=True)
    print("=============================================\n", flush=True)


if __name__ == "__main__":
    # For SLURM: run with `python -u eego_models_binary_separate_losocv_leftoutval.py`
    main()