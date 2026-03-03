# experiment_muse.py
from __future__ import annotations

import time
import threading
import uuid
import pandas as pd
import eegproc

from .Muse.MuseService import start_muse_reader, stop_muse_reader, pow_data_batch, sensor_contact_quality, MuseLSLConfig
from .EpocX.EpocXData import save_eeg_data, featurize_cur_sesh_psd, predict_flow

global_session_id: str | None = None


def set_global_session_id():
    global global_session_id
    global_session_id = str(uuid.uuid4())
    print(f"Initialized global_session_id: {global_session_id}")


def get_global_session_id() -> str:
    global global_session_id
    if not global_session_id:
        raise ValueError("Global session_id is not set.")
    return global_session_id


def init_muse_record():
    """
    Starts the Muse LSL reader in a daemon thread.
    IMPORTANT: MuseLSL must already be streaming EEG into LSL.
      - Typical: run `muselsl stream` in a terminal (or BlueMuse on Windows)
    """
    cfg = MuseLSLConfig(
        stream_type="EEG",
        stream_name=None,   # set if you want exact name match
        window_sec=1.0,
        hop_sec=0.25,
        max_resolve_sec=10,
    )
    t = start_muse_reader(cfg)
    return t


def set_session_id():
    set_global_session_id()
    t = threading.Thread(target=init_muse_record, daemon=True)
    t.start()
    time.sleep(2)  # give stream a moment to resolve / warm up
    return


def predict_n_insert(
    user_id: int,
    object_count: int,
    time_elapsed: float,
    arousal: int,
    valence: int,
    fall_speed: float,
    difficulty_type: str,
):
    """
    Mirrors your EpocX pipeline, but uses MuseService globals.
    """
    check_tick_time = time.time()
    session_id = get_global_session_id()

    # Snapshot current batch so featurization / prediction isn't racing the reader
    batch_snapshot = pow_data_batch.copy()

    featurized_batch = featurize_cur_sesh_psd(
        user_id,
        session_id,
        object_count,
        time_elapsed,
        arousal,
        valence,
        fall_speed,
        difficulty_type,
        float(sensor_contact_quality),
        batch_snapshot,
    )

    arousal_pred, valence_pred = predict_flow(featurized_batch)

    save_eeg_data(
        "dreamer_models/datasets/curr_sesh_muse.csv",
        user_id,
        session_id,
        object_count,
        time_elapsed,
        arousal_pred,
        valence_pred,
        fall_speed,
        difficulty_type,
        float(sensor_contact_quality),
        batch_snapshot,
    )

    # Clear the shared global batch (like your EpocX code does)
    pow_data_batch.drop(pow_data_batch.index, inplace=True)

    print("Prediction time:", time.time() - check_tick_time)
    return (arousal_pred, valence_pred)


def save_curr_sesh(path_a: str, path_b: str) -> pd.DataFrame:
    df_a = pd.read_csv(path_a)
    cols_a = df_a.columns.tolist()
    df_b = pd.read_csv(path_b, usecols=cols_a)
    combined = pd.concat([df_a, df_b[cols_a]], ignore_index=True)
    combined.to_csv(path_a, index=False)
    return combined


def shutdown_muse():
    stop_muse_reader()