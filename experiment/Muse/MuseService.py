# MuseService.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_byprop


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
# Bandpower helpers
# ============================================================

def _welch_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal Welch PSD without scipy dependency if needed.
    If you DO have scipy, you can swap this with scipy.signal.welch.
    """
    # Simple periodogram-style PSD (not as good as Welch, but robust / dependency-free).
    # For better PSD, prefer scipy.signal.welch.
    n = len(x)
    if n < 8:
        freqs = np.array([0.0])
        psd = np.array([0.0])
        return freqs, psd

    x = x - np.mean(x)
    win = np.hanning(n)
    xw = x * win
    fft = np.fft.rfft(xw)
    psd = (np.abs(fft) ** 2) / (fs * np.sum(win ** 2))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, psd


def _bandpower(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    if not np.any(idx):
        return 0.0
    # integrate PSD over band
    return float(np.trapz(psd[idx], freqs[idx]))


def _compute_band_powers_window(window: np.ndarray, fs: float) -> Dict[str, Dict[str, float]]:
    """
    window: shape (n_samples, n_channels) in CHANNELS order
    returns: {channel: {band: power}}
    """
    # Band definitions (Hz) to match your theta/alpha/betaL/betaH/gamma
    band_defs = {
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "betaL": (13.0, 20.0),
        "betaH": (20.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    out: Dict[str, Dict[str, float]] = {}
    for ci, ch in enumerate(CHANNELS):
        x = window[:, ci].astype(np.float64, copy=False)
        freqs, psd = _welch_psd(x, fs)

        out[ch] = {}
        for band, (fmin, fmax) in band_defs.items():
            out[ch][band] = _bandpower(freqs, psd, fmin, fmax)

    return out


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
        fs = 256.0

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
            bp = _compute_band_powers_window(window, fs)

            row: List[float] = []
            for ch in CHANNELS:
                for band in BANDS:
                    row.append(bp[ch][band])

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