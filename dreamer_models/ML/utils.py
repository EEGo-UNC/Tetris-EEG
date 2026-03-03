from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Union, Set
from typing import Iterable, List, Dict
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import chain, combinations, product
import re

_PSD_RE = re.compile(
    r"^(?P<ch>[A-Z0-9]+)_(?P<band>delta|theta|alpha|betaL|betaH|gamma)$"
)
_ENT_RE = re.compile(r"^(?P<ch>[A-Z0-9]+)_entropy$")
_PAIR_RE = re.compile(
    r"^(?P<ch1>[A-Z0-9]+)_(?P<ch2>[A-Z0-9]+)_(?P<band>delta|theta|alpha|betaL|betaH|gamma)_(?P<kind>da|ra)$"
)


def read_table(filename: str = "datasets/features_table.csv") -> pd.DataFrame:
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir.parent / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    df = pd.read_csv(file_path)
    return df


def generate_all_subsets(
    columns: Iterable[str],
    *,
    sensor_families: Optional[Iterable[str]] = None,
    feature_tokens: Optional[Iterable[str]] = None,
    freq_tokens: Optional[Iterable[str]] = None,
    min_size: int = 6,
) -> List[List[str]]:
    if sensor_families is None:
        sensor_families = []
    if feature_tokens is None:
        feature_tokens = ["entropy", "", "da", "ra"]
    if freq_tokens is None:
        freq_tokens = ["theta", "beta", "alpha", "delta", "gamma"]

    sensor_families = list(sensor_families)
    feature_tokens_sorted = sorted(set(feature_tokens), key=len, reverse=True)
    freq_tokens_sorted = sorted(set(freq_tokens), key=len, reverse=True)

    leading_chan_re = re.compile(r"^(?P<prefix>[A-Za-z]+)[0-9A-Za-z]*", re.ASCII)

    def infer_sensor_family(col: str) -> Optional[str]:
        # Try channel-like prefix first (e.g., AF3 -> AF)
        m = leading_chan_re.match(col)
        if m:
            prefix = m.group("prefix").upper()
            for fam in sorted(sensor_families, key=len, reverse=True):
                if prefix.startswith(fam.upper()):
                    return fam
        # Fallback: substring search
        up = col.upper()
        for fam in sorted(sensor_families, key=len, reverse=True):
            if fam.upper() in up:
                return fam
        return None

    def infer_freq(col: str) -> Optional[str]:
        low = col.lower()
        for f in freq_tokens_sorted:
            if f.lower() in low:
                return f
        return None

    def infer_feature(col: str) -> str:
        low = col.lower()
        for tok in feature_tokens_sorted:
            if tok.lower() in low:
                return tok
        return "psd"

    parsed: Dict[str, Tuple[Optional[str], str, Optional[str]]] = {}
    for c in columns:
        parsed[c] = (infer_sensor_family(c), infer_feature(c), infer_freq(c))

    present_fams = sorted({fam for fam, _, _ in parsed.values() if fam is not None})
    present_feats = sorted({feat for _, feat, _ in parsed.values() if feat is not None})
    present_freqs = sorted({fr for _, _, fr in parsed.values() if fr is not None})

    def powerset_with_empty(items: List[str]):
        s = list(items)
        return chain.from_iterable(combinations(s, r) for r in range(0, len(s) + 1))

    fam_subsets = list(powerset_with_empty(present_fams))
    feat_subsets = list(powerset_with_empty(present_feats))
    freq_subsets = list(powerset_with_empty(present_freqs))

    original_order = list(columns)
    all_subsets: List[List[str]] = []
    seen: set = set()

    for fam_sel, feat_sel, freq_sel in product(fam_subsets, feat_subsets, freq_subsets):
        fam_set = set(fam_sel) if len(fam_sel) else None
        feat_set = set(feat_sel) if len(feat_sel) else None
        freq_set = set(freq_sel) if len(freq_sel) else None

        selected = []
        for c in original_order:
            fam, feat, fr = parsed[c]
            if fam_set is not None and fam not in fam_set:
                continue
            if feat_set is not None and feat not in feat_set:
                continue
            if freq_set is not None and fr not in freq_set:
                continue
            selected.append(c)

        if len(selected) >= min_size:
            key = tuple(selected)
            if key not in seen:
                seen.add(key)
                all_subsets.append(selected)

    return all_subsets


def important_features_list(
    filename: str = "datasets/USE_feature_importance/dreamer_feature_importance_arousal.csv",
):
    df = pd.read_csv(filename)

    return df["feature"].to_list()


def filter_features(
    features: Iterable[str],
    *,
    remove_channels: Optional[Iterable[str]] = None,  # e.g., ["AF3","F7"]
    remove_bands: Optional[Iterable[str]] = None,  # e.g., ["gamma","betaH"]
    remove_types: Optional[
        Iterable[str]
    ] = None,  # any of {"psd","entropy","da","ra","pair","pair_da","pair_ra"}
    remove_pairs_with_channels: Optional[
        Iterable[str]
    ] = None,  # drop any pair feature involving any of these channels
) -> List[str]:
    chans: Set[str] = set(map(str.upper, remove_channels or []))
    pbands: Set[str] = set(remove_bands or [])
    ptypes: Set[str] = set((remove_types or []))
    pair_chans: Set[str] = set(map(str.upper, remove_pairs_with_channels or []))

    def should_drop(feat: str) -> bool:
        f = feat.strip()

        m = _PAIR_RE.match(f)
        if m:
            ch1, ch2 = m["ch1"].upper(), m["ch2"].upper()
            band, kind = m["band"], m["kind"]  # kind in {"da","ra"}

            if ch1 in chans or ch2 in chans:
                return True
            if pair_chans and (ch1 in pair_chans or ch2 in pair_chans):
                return True
            if band in pbands:
                return True
            if "pair" in ptypes:
                return True
            if kind in ptypes or f"pair_{kind}" in ptypes:
                return True
            return False

        # Entropy: CH_entropy
        m = _ENT_RE.match(f)
        if m:
            ch = m["ch"].upper()
            if ch in chans:
                return True
            if "entropy" in ptypes:
                return True
            return False

        # Single-channel PSD: CH_band
        m = _PSD_RE.match(f)
        if m:
            ch, band = m["ch"].upper(), m["band"]
            if ch in chans:
                return True
            if band in pbands:
                return True
            if "psd" in ptypes:
                return True
            return False

        # If it doesn't match known schemas, keep it.
        return False

    return [f for f in features if not should_drop(f)]


# Homologous left–right pairs for Emotiv/DREAMER (14ch)
HOMOLOGOUS_PAIRS = [
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC5", "FC6"),
    ("T7", "T8"),
    ("P7", "P8"),
    ("O1", "O2"),
]


AS_BANDS = {"delta", "theta", "alpha", "beta", "gamma"}


def compute_asymmetry_from_psd(
    psd: pd.DataFrame,
    pairs: list[tuple[str, str]] = HOMOLOGOUS_PAIRS,
    eps: float = 1e-12,
    add_log: bool = True,
    prefix_da: str = "da",
    prefix_ra: str = "ra",
) -> pd.DataFrame:
    bands_present = set()
    for col in psd.columns:
        if "_" in col:
            ch, band = col.rsplit("_", 1)
            if band in AS_BANDS:
                bands_present.add(band)

    out_cols = {}

    for L, R in pairs:
        for band in bands_present:
            cL = f"{L}_{band}"
            cR = f"{R}_{band}"
            if cL not in psd.columns or cR not in psd.columns:
                continue

            PL = psd[cL].astype(float)
            PR = psd[cR].astype(float)

            if add_log:
                da = np.log(PR + eps) - np.log(PL + eps)
            else:
                da = (PR + eps) - (PL + eps)

            ra = (PR - PL) / (PR + PL + eps)

            out_cols[f"{R}_{L}_{band}_{prefix_da}"] = da
            out_cols[f"{R}_{L}_{band}_{prefix_ra}"] = ra

    return pd.DataFrame(out_cols, index=psd.index)


def plot_regressor_accuracy(y_true, y_pred, size_increment=5, title=None):

    fig, ax = plt.subplots(figsize=(5, 5))

    size: dict[tuple, int] = {}
    for i in range(len(y_true)):
        if (y_pred[i], y_true[i]) in size:
            size[(y_pred[i], y_true[i])] += size_increment
        else:
            size[(y_pred[i], y_true[i])] = size_increment

    s = []
    for i in range(len(y_true)):
        s.append(size[(y_pred[i], y_true[i])])

    ax.scatter(y_true, y_pred, s=s, alpha=0.7, edgecolors="none")
    ax.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], c="blue", linestyle="--", linewidth=2)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    if title:
        ax.set_title(title)

    return ax


if __name__ == "__main__":
    important_features_list(
        "dreamer_models/datasets/USE_feature_importance/dreamer_feature_importance_arousal.csv"
    )
