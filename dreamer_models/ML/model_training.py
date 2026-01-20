import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.optimizers import Adam
from .STSNet import STSNetModel

from .labels import *
from .splits import *


def train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    estimators: int = 100,
    max_depth: int = 8,
    n_jobs: int = -1,
):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, verbose=True
    )
    rf.fit(X_train, y_train)

    return rf, X_test, y_test


def train_random_forest_regressor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    estimators: int = 100,
    max_depth: int = 8,
    n_jobs: int = -1,
):
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, verbose=True
    )
    rf.fit(X_train, y_train)

    return rf, X_test, y_test


def train_knn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    neighbors: int = 1,
    weights: str = "distance",
    n_jobs: int = -1,
):
    knn = KNeighborsClassifier(
        n_neighbors=neighbors, weights=weights, n_jobs=n_jobs, p=1
    )
    knn.fit(X_train, y_train)

    return knn, X_test, y_test


def train_knn_regressor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    neighbors: int = 1,
    weights: str = "distance",
    n_jobs: int = -1,
):
    knn = KNeighborsRegressor(
        n_neighbors=neighbors, weights=weights, n_jobs=n_jobs, p=1
    )
    knn.fit(X_train, y_train)
    return knn, X_test, y_test


def build_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "arousal",
    thresh: float = 3.8,
    fixed_T: int | None = None,  # e.g. 15 windows for 60s if you want
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (num_trials, timesteps, n_features) and (num_trials,) from a window-level DF.

    Each trial = (patient_index, video_index).
    Rows for each trial must be contiguous and in temporal order.
    """
    # Group by trial; we rely on the existing row order for time
    groups = df.groupby(["patient_index", "video_index"], sort=False)

    X_seqs: list[np.ndarray] = []
    y_labels: list[float] = []

    for (_, _), g in groups:
        # (Ti, n_features) for this trial
        X_seq = g[feature_cols].to_numpy(dtype=np.float32)

        # trial-level label (same for all windows in that trial)
        y_val = g[target_col].iloc[0]
        y_bin = 1.0 if y_val > thresh else 0.0

        X_seqs.append(X_seq)
        y_labels.append(y_bin)

    if fixed_T is None:
        fixed_T = max(seq.shape[0] for seq in X_seqs)

    n_features = X_seqs[0].shape[1]
    X_padded = np.zeros((len(X_seqs), fixed_T, n_features), dtype=np.float32)

    for i, seq in enumerate(X_seqs):
        T = seq.shape[0]
        if T >= fixed_T:
            X_padded[i, :, :] = seq[:fixed_T, :]
        else:
            X_padded[i, :T, :] = seq

    y_arr = np.asarray(y_labels, dtype=np.float32)
    return X_padded, y_arr


def build_eego_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "affect_arousal",  # or "affect_valence", "arousal", etc.
    thresh: float = 3.0,
    fixed_T: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (num_trials, timesteps, n_features) and (num_trials,) from EEGo
    window-level data.

    Each trial = (user_id, session_id, affect_minute).

    Rows for each trial should correspond to contiguous windows in time,
    sorted by time_elapsed (or timestamp).
    """
    groups = df.groupby(["user_id", "session_id", "affect_minute"], sort=False)

    X_seqs: list[np.ndarray] = []
    y_labels: list[float] = []

    for (_, _, _), g in groups:
        # if "time_elapsed" in g.columns:
        #     g = g.sort_values("time_elapsed")
        # elif "timestamp" in g.columns:
        #     g = g.sort_values("timestamp")

        X_seq = g[feature_cols].to_numpy(dtype=np.float32)

        y_vals = g[target_col].values
        assert np.allclose(y_vals, y_vals[0]), "Label not constant within group!"

        y_val = float(y_vals[0])
        y_bin = 1.0 if y_val > thresh else 0.0

        X_seqs.append(X_seq)
        y_labels.append(y_bin)

    # if fixed_T is None:
    #     fixed_T = max(seq.shape[0] for seq in X_seqs)

    n_features = X_seqs[0].shape[1]
    X_padded = np.zeros((len(X_seqs), fixed_T, n_features), dtype=np.float32)

    for i, seq in enumerate(X_seqs):
        T = seq.shape[0]
        if T >= fixed_T:
            X_padded[i, :, :] = seq[:fixed_T, :]
        else:
            X_padded[i, :T, :] = seq

    y_arr = np.asarray(y_labels, dtype=np.float32)
    return X_padded, y_arr


def train_lstm(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | pd.DataFrame | np.ndarray,
    y_test: pd.Series | pd.DataFrame | np.ndarray,
    *,
    units: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    bidirectional: bool = True,
    patience: int = 8,
    verbose: int = 0,
    random_seed: int = 42,
):
    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)

    X_train_arr = np.asarray(X_train).astype(np.float32, copy=False)
    X_test_arr = np.asarray(X_test).astype(np.float32, copy=False)
    y_train_arr = np.asarray(y_train)

    # Handle labels flexibly: numeric 0/1 or "high"/"low"
    if y_train_arr.dtype.kind in "ifu":  # int/float
        y_train_arr = y_train_arr.astype(np.float32)
    else:
        y_train_arr = np.array(
            [1.0 if str(v).lower() == "high" else 0.0 for v in y_train_arr],
            dtype=np.float32,
        )

    if y_test is not None:
        y_test_arr = np.asarray(y_test)
        if y_test_arr.dtype.kind in "ifu":
            y_test_arr = y_test_arr.astype(np.float32)
        else:
            y_test_arr = np.array(
                [1.0 if str(v).lower() == "high" else 0.0 for v in y_test_arr],
                dtype=np.float32,
            )
    else:
        y_test_arr = None

    if X_train_arr.ndim != 3:
        raise ValueError(
            f"X_train must be 3D (trials, timesteps, features), got {X_train_arr.shape}"
        )

    timesteps = X_train_arr.shape[1]
    n_features = X_train_arr.shape[2]

    inp = layers.Input(shape=(timesteps, n_features))
    lstm_block = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
    )
    x = layers.Bidirectional(lstm_block)(inp) if bidirectional else lstm_block(inp)
    # lstm_block2 = layers.LSTM(
    #     units,
    #     dropout=dropout,
    #     recurrent_dropout=recurrent_dropout,
    #     return_sequences=False,
    # )
    # x = layers.Bidirectional(lstm_block2)(x) if bidirectional else lstm_block2(x)
    x = layers.Dense(units // 2, activation="leaky_relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            mode="max",
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            mode="max",
        ),
    ]

    model.fit(
        X_train_arr,
        y_train_arr,
        epochs=epochs,
        validation_split=0.15,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=cbs,
        shuffle=False,
    )

    return model, X_test_arr, y_test_arr


def train_lstm_regressor(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | pd.DataFrame | np.ndarray,
    y_test: pd.Series | pd.DataFrame | np.ndarray,
    *,
    units: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    bidirectional: bool = True,
    patience: int = 8,
    verbose: int = 0,
    random_seed: int = 42,
):
    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)

    X_train_arr = np.asarray(X_train).astype(np.float32, copy=False)
    X_test_arr = np.asarray(X_test).astype(np.float32, copy=False)

    y_train_arr = np.asarray(y_train, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.float32)

    X_test_arr

    timesteps = X_train_arr.shape[1]
    n_features = X_train_arr.shape[2]

    inp = layers.Input(shape=(timesteps, n_features))
    lstm_block = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
    )
    x = layers.Bidirectional(lstm_block)(inp) if bidirectional else lstm_block(inp)
    x = layers.Dense(units // 2, activation="leaky_relu")(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(1, activation="linear")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ],
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="loss", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-6
        ),
    ]

    model.fit(
        X_train_arr,
        y_train_arr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=cbs,
        shuffle=False,
    )

    return model, X_test_arr, y_test_arr


def STSNet(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    n_channels: int = 14,
    random_seed: int | None = 42,
    epochs: int = 50,
    batch_size: int = 64,
    verbose: int = 0,
    learning_rate: float = 1e-4,
):

    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)
    X_train_arr = np.asarray(X_train).astype(np.float32, copy=False)
    X_test_arr = np.asarray(X_test).astype(np.float32, copy=False)
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    y_train_arr = np.array(
        [1 if str(v).lower() == "high" else 0 for v in y_train_arr],
        dtype=np.int32,
    )
    y_test_arr = np.array(
        [1 if str(v).lower() == "high" else 0 for v in y_test_arr],
        dtype=np.int32,
    )

    if X_train_arr.ndim != 2:
        raise ValueError(
            f"Expected X_train to be 2D (N, features), got shape {X_train_arr.shape}"
        )

    total_features = X_train_arr.shape[1]
    if total_features % n_channels != 0:
        raise ValueError(
            f"total_features={total_features} is not divisible by n_channels={n_channels}. "
            "Check your feature ordering / channel count."
        )

    n_features = total_features // n_channels

    X_train_arr = X_train_arr.reshape(-1, n_channels, n_features)
    X_test_arr = X_test_arr.reshape(-1, n_channels, n_features)

    X_train_arr = X_train_arr[..., np.newaxis]
    X_test_arr = X_test_arr[..., np.newaxis]

    model = STSNetModel(
        n_classes=2,
        n_channels=n_channels,
        n_features=n_features,
    )

    opt = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    model.fit(
        X_train_arr,
        y_train_arr,
        validation_data=(X_test_arr, y_test_arr),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    return model, X_test_arr, y_test_arr
