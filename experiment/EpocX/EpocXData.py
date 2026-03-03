from datetime import datetime
import pandas as pd
import os
import eegproc as eeg
from dreamer_models.ML.utils import compute_asymmetry_from_psd
from dreamer_models.predictor_model import arousal_model, valence_model, features
import numpy as np


FS = 128


def save_eeg_data(
    filename: str,
    user_id: int,
    session_id: int,
    object_count: int,
    time_elapsed: float,
    arousal: int,
    valence: int,
    fall_speed: float,
    difficulty_type: str,
    sensor_contact_quality: bool,
    df: pd.DataFrame,
) -> int:
    if df is None or df.empty:
        return 0

    df_to_write = df.copy()
    df_to_write.insert(0, "user_id", user_id)
    df_to_write.insert(1, "session_id", session_id)
    df_to_write.insert(2, "object_count", object_count)
    df_to_write.insert(3, "time_elapsed", time_elapsed)
    df_to_write.insert(5, "arousal", arousal)
    df_to_write.insert(6, "valence", valence)
    df_to_write.insert(7, "fall_speed", fall_speed)
    df_to_write.insert(8, "difficulty_type", difficulty_type)
    df_to_write.insert(9, "sensor_contact_quality", sensor_contact_quality)

    file_exists = os.path.exists(filename)

    if file_exists:
        existing_cols = pd.read_csv(filename, nrows=0).columns.tolist()

        incoming_cols = df_to_write.columns.tolist()
        if set(existing_cols) != set(incoming_cols):
            missing = set(existing_cols) - set(incoming_cols)
            extra = set(incoming_cols) - set(existing_cols)
            raise ValueError(
                f"Column mismatch.\nMissing in incoming: {missing}\nExtra in incoming: {extra}"
            )

        df_to_write = df_to_write[existing_cols]
        df_to_write.to_csv(
            filename,
            mode="a",
            index=False,
            header=False,
            encoding="utf-8",
            lineterminator="\n",
        )


def featurize_cur_sesh_psd(
    user_id: int,
    session_id: int,
    object_count: int,
    time_elapsed: float,
    arousal: int,
    valence: int,
    fall_speed: float,
    difficulty_type: str,
    sensor_contact_quality: bool,
    df: pd.DataFrame,
) -> pd.DataFrame:
    n = len(df)

    shannons = eeg.shannons_entropy(df)
    asymm = compute_asymmetry_from_psd(df)

    meta = pd.DataFrame(
        {
            "user_id": pd.Series([user_id] * n),
            "session_id": pd.Series([session_id] * n),
            "object_count": pd.Series([object_count] * n),
            "time_elapsed": pd.Series([time_elapsed] * n),
            "arousal": pd.Series([arousal] * n),
            "valence": pd.Series([valence] * n),
            "fall_speed": pd.Series([fall_speed] * n),
            "difficulty_type": pd.Series([difficulty_type] * n),
            "sensor_contact_quality": pd.Series([sensor_contact_quality] * n),
        }
    )
    batch = pd.concat([meta, df, shannons, asymm], axis=1)

    return batch


def predict_flow(featurized_batch: pd.DataFrame) -> int:
    batch = featurized_batch.drop(
        columns=[
            "user_id",
            "session_id",
            "object_count",
            "time_elapsed",
            "arousal",
            "valence",
            "fall_speed",
            "difficulty_type",
            "sensor_contact_quality",
            "timestamp",
        ]
    )

    feature_cols = [c for c in features if c in batch.columns]
    batch = batch.reindex(columns=feature_cols)
    batch = batch.reset_index(drop=True)

    X = batch.to_numpy().astype("float32", copy=False)

    print("Before reshape, X.shape =", X.shape)  # (timesteps, n_features)

    X = batch.to_numpy().astype("float32", copy=False)
    T, F = X.shape

    _, expected_T, expected_F = arousal_model.input_shape
    if F != expected_F:
        raise ValueError(f"Feature mismatch: X has {F}, model expects {expected_F}")

    if T > expected_T:
        X = X[-expected_T:, :]
    elif T < expected_T:
        pad_len = expected_T - T
        pad = np.zeros((pad_len, F), dtype=X.dtype)
        X = np.concatenate([X, pad], axis=0)

    X = np.expand_dims(X, axis=0)

    print("After reshape, X.shape  =", X.shape)  # (1, timesteps, n_features)
    print("Model expects:", arousal_model.input_shape)
    print(X)

    arousal = arousal_model.predict(X).ravel()
    valence = valence_model.predict(X).ravel()
    print(arousal)

    arousal_score = float(np.mean(arousal) * 5)
    valence_score = float(np.mean(valence) * 5)

    print("arousal mean * 5:", arousal_score)
    print("valence mean * 5:", valence_score)

    return arousal_score, valence_score
