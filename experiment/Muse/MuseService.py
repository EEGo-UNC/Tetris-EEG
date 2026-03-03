# MuseService.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_byprop
import eegproc


# ============================================================
# Globals (match EpocXService style)
# ============================================================

# Muse EEG channels typically: TP9, AF7, AF8, TP10
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# Match your EpocX band names
BANDS = ["theta", "alpha", "betaL", "betaH", "gamma"]

cols = [f"{ch}_{band}" for ch in CHANNELS for band in BANDS]
cols.append("timestamp")

pow_data_batch = pd.DataFrame(columns=cols)

# MuseLSL doesn't provide "contact quality" like Cortex dev stream.
# Keep it as a float in [0,1] or a bool; here we just expose a float.
sensor_contact_quality: float = 1.0

# Thread control
_stop_event = threading.Event()
_reader_thread: Optional[threading.Thread] = None



# ============================================================
# LSL reader
# ============================================================

@dataclass
class MuseLSLConfig:
    stream_name: Optional[str] = None     # if you want to force name match
    stream_type: str = "EEG"              # MuseLSL sets type "EEG"
    window_sec: float = 1.0               # bandpower window size
    hop_sec: float = 0.25                 # how often to emit a new row
    max_resolve_sec: int = 10             # wait time to find LSL stream


def _resolve_muse_eeg_stream(cfg: MuseLSLConfig):
    """
    Finds an LSL stream. Default: resolve by type='EEG'
    """
    t0 = time.time()
    while time.time() - t0 < cfg.max_resolve_sec and not _stop_event.is_set():
        if cfg.stream_name:
            streams = resolve_byprop("name", cfg.stream_name, timeout=1.0)
        else:
            streams = resolve_byprop("type", cfg.stream_type, timeout=1.0)
        if streams:
            return streams[0]
    raise RuntimeError(
        f"Could not resolve LSL stream (type={cfg.stream_type}, name={cfg.stream_name}). "
        f"Is MuseLSL streaming? (e.g., run `muselsl stream` or start BlueMuse + LSL)"
    )


def _lsl_reader_loop(cfg: MuseLSLConfig):
    global pow_data_batch, sensor_contact_quality 

    info = _resolve_muse_eeg_stream(cfg)
    inlet = StreamInlet(info, max_buflen=10)

    fs = float(info.nominal_srate())
    if fs <= 0:
        # fallback; Muse 2 is usually 256 Hz
        fs = 128.0

    n_chan = info.channel_count()
    if n_chan < 4:
        raise RuntimeError(f"LSL EEG stream has {n_chan} channels; expected >= 4 for Muse EEG.")

    # We will take the first 4 channels as TP9, AF7, AF8, TP10.
    # (MuseLSL typically streams in that order for EEG.)
    win_n = int(round(cfg.window_sec * fs))
    hop_n = int(round(cfg.hop_sec * fs))
    win_n = max(win_n, 32)
    hop_n = max(hop_n, 1)

    buf = np.zeros((0, 4), dtype=np.float32)
    last_emit_samples = 0

    # If you want a “quality” proxy, you can set it based on amplitude sanity checks.
    # Here: simple heuristic in [0,1].
    def quality_proxy(x: np.ndarray) -> float:
        # very rough: penalize huge spikes / saturation
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms <= 1e-6:
            return 0.0
        if rms > 500.0:  # uV-scale spike
            return 0.2
        return 1.0

    while not _stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue

        s4 = np.array(sample[:4], dtype=np.float32)[None, :]  # (1,4)
        buf = np.concatenate([buf, s4], axis=0)

        # update "contact" proxy
        sensor_contact_quality = quality_proxy(s4)

        # emit at hop rate once we have enough for a window
        if len(buf) >= win_n and (len(buf) - last_emit_samples) >= hop_n:
            window = buf[-win_n:, :]  # (win_n,4)
            window_df = pd.DataFrame(window, columns=CHANNELS)
            bp_filt = eegproc.bandpass_filter(window_df, fs=128, bands=eegproc.FREQUENCY_BANDS, low=5.0, high=45)
            batch_psd = eegproc.psd_bandpowers(bp_filt, fs=128, bands=eegproc.FREQUENCY_BANDS, window_sec=2)
    

            row: List[float] = []
            for ch in CHANNELS:
                for band in BANDS:
                    row.append(float(batch_psd[f"{ch}_{band}"].iloc[-1]))

            # timestamp: use LSL timestamp ts
            row.append(float(ts))

            pow_data_batch.loc[len(pow_data_batch)] = row
            last_emit_samples = len(buf)

            # keep buffer bounded
            max_keep = int(round(10 * fs))
            if len(buf) > max_keep:
                buf = buf[-max_keep:, :]
                last_emit_samples = min(last_emit_samples, len(buf))


def start_muse_reader(cfg: Optional[MuseLSLConfig] = None) -> threading.Thread:
    """
    Starts a daemon thread that fills pow_data_batch continuously.
    """
    global _reader_thread
    if cfg is None:
        cfg = MuseLSLConfig()

    stop_muse_reader()  # reset if already running
    _stop_event.clear()

    _reader_thread = threading.Thread(target=_lsl_reader_loop, args=(cfg,), daemon=True)
    _reader_thread.start()
    return _reader_thread


def stop_muse_reader():
    global _reader_thread
    if _reader_thread and _reader_thread.is_alive():
        _stop_event.set()
        _reader_thread.join(timeout=2.0)
    _reader_thread = None
    _stop_event.clear()