import os
import asyncio
from .EpocX.EpocXData import save_eeg_data, featurize_cur_sesh_psd, predict_flow
from .EpocX import EpocXService as EpocX
import pandas as pd
import time
import multiprocessing as mp
import threading
import uuid

global_session_id: str = None


def set_global_session_id():
    """Initialize the global session ID."""
    global global_session_id
    global_session_id = str(uuid.uuid4())
    print(f"Initialized global session_id: {global_session_id}")


def get_global_session_id():
    """Retrieve the global session ID."""
    global global_session_id
    if not global_session_id:
        raise ValueError("Global session_id is not set.")
    return global_session_id


async def set_session_id():
    start = time.time()
    set_global_session_id()
    t = threading.Thread(target=init_epoc_record)
    t.start()
    time.sleep(3)
    print(time.time() - start)
    return


def init_epoc_record():
    def runner():
        asyncio.run(EpocX.main())

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t


def predict_n_insert(
    user_id: int,
    object_count: int,
    time_elapsed: float,
    arousal: int,
    valence: int,
    fall_speed: float,
    difficulty_type: str,
):
    check_tick_time = time.time() 
    session_id = get_global_session_id()
    featurized_batch = featurize_cur_sesh_psd(
        user_id,
        session_id,
        object_count,
        time_elapsed,
        arousal,
        valence,
        fall_speed,
        difficulty_type,
        EpocX.sensor_contact_quality,
        EpocX.pow_data_batch,
    )
    arousal, valence = predict_flow(featurized_batch)
    save_eeg_data(
        "dreamer_models/datasets/curr_sesh.csv",
        user_id,
        session_id,
        object_count,
        time_elapsed,
        arousal,
        valence,
        fall_speed,
        difficulty_type,
        EpocX.sensor_contact_quality,
        EpocX.pow_data_batch,
    )
    
    EpocX.pow_data_batch.drop(EpocX.pow_data_batch.index, inplace=True)
    print("Prediction time:", time.time() - check_tick_time)

    return (arousal, valence)


def save_curr_sesh(path_a: str, path_b: str) -> pd.DataFrame:
    df_a = pd.read_csv(path_a)
    cols_a = df_a.columns.tolist()

    df_b = pd.read_csv(path_b, usecols=cols_a)

    # Concatenate row-wise
    combined = pd.concat([df_a, df_b[cols_a]], ignore_index=True)

    combined.to_csv(path_a, index=False)

    return combined
