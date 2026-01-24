from __future__ import annotations

import os
import time
from pathlib import Path
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


# ============================================================
#                     Data prep helpers
# ============================================================
def load_eego_df(filename: str | Path = "EEGo_labeled.csv") -> pd.DataFrame:
    df = pd.read_csv(filename)

    sort_cols = ["user_id", "session_id"]
    if "time_elapsed" in df.columns:
        sort_cols.append("time_elapsed")
    elif "timestamp" in df.columns:
        sort_cols.append("timestamp")

    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def select_eego_features(df: pd.DataFrame) -> list[str]:
    eeg_prefixes = [
        "AF3_", "F7_", "F3_", "FC5_", "T7_", "P7_",
        "O1_", "O2_", "P8_", "T8_", "FC6_", "F4_", "F8_", "AF4_",
    ]
    eeg_features = [c for c in df.columns if any(c.startswith(p) for p in eeg_prefixes)]
    return eeg_features


# ============================================================
#              4-class (Valence-Arousal) labeling
# ============================================================
# Class ordering:
# 0: LVLA (low valence, low arousal)
# 1: LVHA (low valence, high arousal)
# 2: HVLA (high valence, low arousal)
# 3: HVHA (high valence, high arousal)
#
# class_id = 2 * v_bin + a_bin
def build_eego_lstm_sequences_4class(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    valence_col: str = "affect_valence",
    arousal_col: str = "affect_arousal",
    thresh: float = 2.5,
    fixed_T: int = 2500,
    group_cols: tuple[str, str] = ("user_id", "session_id"),
    seq_label_mode: str = "majority",  # "majority" or "last"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds one fixed-length sequence per (user_id, session_id).

    Returns:
      X_seq: (N, fixed_T, n_features)
      y_seq: (N,) int in {0,1,2,3}
    """
    work = df.copy()

    # Sort within session
    sort_cols = list(group_cols)
    if "time_elapsed" in work.columns:
        sort_cols.append("time_elapsed")
    elif "timestamp" in work.columns:
        sort_cols.append("timestamp")
    work = work.sort_values(sort_cols).reset_index(drop=True)

    # Quadrant label per-row
    v_bin = (work[valence_col].to_numpy() >= thresh).astype(np.int32)
    a_bin = (work[arousal_col].to_numpy() >= thresh).astype(np.int32)
    work["_va4"] = (2 * v_bin + a_bin).astype(np.int32)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for _, g in work.groupby(list(group_cols), sort=False):
        X = g[feature_cols].to_numpy(dtype=np.float32)
        y = g["_va4"].to_numpy(dtype=np.int32)

        # Fixed length via trunc/pad
        if len(X) >= fixed_T:
            X_fix = X[:fixed_T]
            y_fix = y[:fixed_T]
        else:
            pad = fixed_T - len(X)
            X_fix = np.pad(X, ((0, pad), (0, 0)), mode="constant")
            # For labels, pad by edge so last known label repeats
            y_fix = np.pad(y, (0, pad), mode="edge")

        # Sequence label
        if seq_label_mode == "last":
            seq_label = int(y_fix[-1])
        else:
            # majority vote
            seq_label = int(np.bincount(y_fix, minlength=4).argmax())

        X_list.append(X_fix)
        y_list.append(seq_label)

    X_seq = np.stack(X_list, axis=0)
    y_seq = np.asarray(y_list, dtype=np.int32)
    return X_seq, y_seq


# ============================================================
#                    Metrics helpers
# ============================================================
def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels=(0, 1, 2, 3),
) -> tuple[dict[str, float], np.ndarray]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return metrics, cm


def summarize_metric_dicts(dicts: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """
    For a list of metric dicts, returns {metric: (mean, std)} (nan-safe).
    """
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
#                 4-class LSTM trainer (softmax)
# ============================================================
def train_lstm_4class(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    lr: float,
    epochs: int,
    units: int,
    batch_size: int,
    patience: int,
    bidirectional: bool = True,
    verbose: int = 0,
) -> tuple[tf.keras.Model, np.ndarray, np.ndarray]:
    """
    Multiclass LSTM with sparse_categorical_crossentropy.
    Uses validation_split internally and EarlyStopping on val_loss.
    """
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (N,T,F). Got shape {X_train.shape}")
    if X_test.ndim != 3:
        raise ValueError(f"X_test must be 3D (N,T,F). Got shape {X_test.shape}")

    n_features = int(X_train.shape[-1])
    T = int(X_train.shape[1])
    n_classes = 4

    inp = tf.keras.Input(shape=(T, n_features))
    x = inp

    if bidirectional:
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units))(x)
    else:
        x = tf.keras.layers.LSTM(units)(x)

    x = tf.keras.layers.Dropout(0.30)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train.astype(np.int32),
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=True,
        callbacks=callbacks,
    )

    return model, X_test, y_test


# ============================================================
#            Final training / saving best model
# ============================================================
def retrain_and_save_best_model_4class(
    features_table: pd.DataFrame,
    feature_cols: list[str],
    *,
    valence_col: str,
    arousal_col: str,
    thresh: float,
    fixed_T: int,
    best_params: dict,
    out_dir: str = "models",
    seq_label_mode: str = "majority",
) -> str:
    """
    Retrain one final 4-class model on ALL sessions using best hyperparams,
    then save as .keras. Returns saved path.
    """
    os.makedirs(out_dir, exist_ok=True)
    full_df = features_table.drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)

    print("\n==============================", flush=True)
    print("FINAL TRAIN | Target: VA-4CLASS", flush=True)
    print("==============================", flush=True)

    # Show quadrant counts at row level (informational)
    v_bin = (full_df[valence_col] >= thresh).astype(int)
    a_bin = (full_df[arousal_col] >= thresh).astype(int)
    full_df["_va4_row"] = 2 * v_bin + a_bin
    print("Row-level VA4 counts:", flush=True)
    print(full_df["_va4_row"].value_counts().sort_index(), flush=True)

    X_seq, y_seq = build_eego_lstm_sequences_4class(
        full_df,
        feature_cols=feature_cols,
        valence_col=valence_col,
        arousal_col=arousal_col,
        thresh=thresh,
        fixed_T=fixed_T,
        seq_label_mode=seq_label_mode,
    )

    # Train (X_seq as both train/test placeholders; fit uses validation_split)
    model, _, _ = train_lstm_4class(
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
        verbose=0,
    )

    save_path = os.path.join(out_dir, "best_va4_lstm.keras")
    model.save(save_path)
    print(f"Saved best final 4-class model -> {save_path}", flush=True)
    return save_path


# ============================================================
#                         Main
# ============================================================
def main() -> None:
    print("GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    print("Starting EEGo LSTM VA-4CLASS CV script...", flush=True)

    # Load EEGo data (labels)
    df_eego = load_eego_df("dreamer_models/datasets/EEGo_labeled.csv")
    print("EEGo shape:", df_eego.shape, flush=True)

    AROUSAL = "affect_arousal"
    VALENCE = "affect_valence"
    THRESH = 2.5

    session_ids = df_eego["session_id"].unique().tolist()
    print("Number of sessions:", len(session_ids), flush=True)

    # Load features table
    features_table = pd.read_csv("dreamer_models/datasets/eego_features.csv")
    feature_cols = select_eego_features(features_table)
    print("n_features:", len(feature_cols), flush=True)

    # Quick 4-class baseline (row-level, just to see imbalance)
    tmp = features_table.copy()
    tmp = tmp.drop(columns=["Unnamed: 0"], errors="ignore")
    v_bin = (tmp[VALENCE] >= THRESH).astype(int)
    a_bin = (tmp[AROUSAL] >= THRESH).astype(int)
    tmp["va4"] = 2 * v_bin + a_bin
    print("Row-level VA4 distribution:", flush=True)
    print(tmp["va4"].value_counts(normalize=True).sort_index(), flush=True)
    print("Majority-class baseline:", float(tmp["va4"].value_counts(normalize=True).max()), flush=True)

    # ---------------- Hyper-parameter grid ----------------
    param_grid = {
        "lr": [1e-4],
        "epochs": [100],
        "units": [256, 512],
        "batch_size": [64, 128],
        "patience": [20],
        "fixed_T": [2500],
        # choose how to label each session-sequence:
        # "majority" = majority label over timesteps
        # "last" = label of final timestep
        "seq_label_mode": ["majority"],
    }

    # ---------------- Folds (leave-k-out style) ----------------
    rng = np.random.default_rng(42)
    rng.shuffle(session_ids)

    # Example: 5 folds (edit as you like)
    folds = np.array_split(session_ids, 5)

    for i, test_sessions in enumerate(folds):
        print(f"Fold {i}: test(k={len(test_sessions)})={sorted(test_sessions.tolist())}", flush=True)

    best_params: dict | None = None
    best_mean_score = -np.inf

    # For imbalance, bal_acc or f1_macro is usually better than raw acc
    OPT_METRIC = "bal_acc"

    for lr, epochs, units, batch_size, patience, fixed_T, seq_label_mode in product(
        param_grid["lr"],
        param_grid["epochs"],
        param_grid["units"],
        param_grid["batch_size"],
        param_grid["patience"],
        param_grid["fixed_T"],
        param_grid["seq_label_mode"],
    ):
        print("\n#############################################", flush=True)
        print(
            f"HP COMBO: lr={lr}, epochs={epochs}, units={units}, batch={batch_size}, "
            f"patience={patience}, fixed_T={fixed_T}, seq_label_mode={seq_label_mode}",
            flush=True,
        )
        print("#############################################", flush=True)

        combo_scores: list[dict[str, float]] = []

        t0_combo = time.time()

        for fold_i, test_sessions in enumerate(folds):
            test_sessions_list = test_sessions.tolist()
            print("\n==============================", flush=True)
            print(f"Fold {fold_i} | LEFT OUT SESSIONS: {test_sessions_list}", flush=True)
            print("==============================", flush=True)

            use_features = features_table.drop(columns=["Unnamed: 0"], errors="ignore")

            mask_test = use_features["session_id"].isin(test_sessions_list)
            train_df = use_features[~mask_test].reset_index(drop=True)
            test_df = use_features[mask_test].reset_index(drop=True)

            # Build 4-class sequences
            X_train_seq, y_train_seq = build_eego_lstm_sequences_4class(
                train_df,
                feature_cols=feature_cols,
                valence_col=VALENCE,
                arousal_col=AROUSAL,
                thresh=THRESH,
                fixed_T=fixed_T,
                seq_label_mode=seq_label_mode,
            )
            X_test_seq, y_test_seq = build_eego_lstm_sequences_4class(
                test_df,
                feature_cols=feature_cols,
                valence_col=VALENCE,
                arousal_col=AROUSAL,
                thresh=THRESH,
                fixed_T=fixed_T,
                seq_label_mode=seq_label_mode,
            )

            print("Train VA4 counts:", np.bincount(y_train_seq, minlength=4), flush=True)
            print("Test  VA4 counts:", np.bincount(y_test_seq, minlength=4), flush=True)

            # Train
            model, X_test_eval, y_test_eval = train_lstm_4class(
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
                verbose=0,
            )

            # Predict
            y_prob = model.predict(X_test_eval, batch_size=batch_size, verbose=0)  # (N,4)
            y_pred = np.argmax(y_prob, axis=1).astype(np.int32)
            y_true = y_test_eval.astype(np.int32)

            fold_metrics, cm = compute_multiclass_metrics(y_true=y_true, y_pred=y_pred)

            print("\nFold metrics (VA-4CLASS):", flush=True)
            for k, v in fold_metrics.items():
                print(f"  {k}: {v:.4f}", flush=True)

            print("Confusion matrix (rows=true, cols=pred):\n", cm, flush=True)
            print("Classification report:", flush=True)
            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=[0, 1, 2, 3],
                    target_names=["LVLA", "LVHA", "HVLA", "HVHA"],
                    digits=4,
                    zero_division=0,
                ),
                flush=True,
            )

            combo_scores.append(fold_metrics)

        # Summaries
        summary = summarize_metric_dicts(combo_scores)
        mean_score = summary.get(OPT_METRIC, (np.nan, np.nan))[0]

        dt_combo = time.time() - t0_combo

        print("\n=============================================", flush=True)
        print(
            f"HP SUMMARY: lr={lr}, epochs={epochs}, units={units}, batch={batch_size}, "
            f"patience={patience}, fixed_T={fixed_T}, seq_label_mode={seq_label_mode}",
            flush=True,
        )
        print(f"Optimizing metric: {OPT_METRIC}", flush=True)
        print(f"Combo runtime: {dt_combo/60:.1f} minutes", flush=True)
        print_summary_block("  VA-4CLASS (mean/std):", summary)
        print(f"  Mean {OPT_METRIC} score = {mean_score:.4f}", flush=True)
        print("=============================================\n", flush=True)

        if mean_score > best_mean_score:
            best_mean_score = float(mean_score)
            best_params = {
                "lr": lr,
                "epochs": epochs,
                "units": units,
                "batch_size": batch_size,
                "patience": patience,
                "fixed_T": fixed_T,
                "seq_label_mode": seq_label_mode,
                "opt_metric": OPT_METRIC,
                "combined_mean_score": float(mean_score),
                "summary": {k: {"mean": m, "std": s} for k, (m, s) in summary.items()},
            }

            print(">>> NEW BEST HP FOUND <<<", flush=True)
            print("Best score so far:", f"{best_mean_score:.4f}", flush=True)
            print("Best params:", best_params, flush=True)

    # Final retrain/save
    if best_params is not None:
        saved_path = retrain_and_save_best_model_4class(
            features_table=features_table,
            feature_cols=feature_cols,
            valence_col=VALENCE,
            arousal_col=AROUSAL,
            thresh=THRESH,
            fixed_T=int(best_params["fixed_T"]),
            best_params=best_params,
            out_dir="models",
            seq_label_mode=str(best_params["seq_label_mode"]),
        )
        print("\nSaved final model path:", saved_path, flush=True)
    else:
        print("No best_params found; skipping final retrain/save.", flush=True)

    print("\n=============================================", flush=True)
    print("Best EEGo VA-4CLASS score:", f"{best_mean_score:.4f}", flush=True)
    print("Best params:", best_params, flush=True)
    print("=============================================\n", flush=True)


if __name__ == "__main__":
    # For SLURM: run with `python -u EEGo_models_va4.py`
    main()
