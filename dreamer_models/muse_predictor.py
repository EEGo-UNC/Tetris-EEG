from pathlib import Path
import numpy as np
import pandas as pd
from .ML.model_training import train_lstm, build_eego_lstm_sequences
import os
from tensorflow.keras.models import load_model


def rename_epocx_to_muse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    QUICK + DIRTY mapping (as requested):
      EpocX F7  -> Muse AF7
      EpocX F8  -> Muse AF8
      EpocX P7  -> Muse TP9
      EpocX P8  -> Muse TP10

    This renames *all* feature columns that start with these prefixes, e.g.
      F7_theta -> AF7_theta
      P8_shannon -> TP10_shannon
      P7_P8_shannon_asym -> TP9_TP10_shannon_asym  (both sides renamed)
    """
    mapping = {
        "F7_": "AF7_",
        "F8_": "AF8_",
        "P7_": "TP9_",
        "P8_": "TP10_",
    }

    rename_map = {}
    for c in df.columns:
        new_c = c
        # Replace any occurrences (not only startswith) so asymmetry cols get fixed too
        for old, new in mapping.items():
            new_c = new_c.replace(old, new)
        if new_c != c:
            rename_map[c] = new_c

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def select_eego_features(df: pd.DataFrame) -> list[str]:
    """
    After renaming, ONLY keep Muse-style prefixes for 4 sensors:
      TP9_, AF7_, AF8_, TP10_
    plus any derived features sharing those prefixes.
    """
    muse_prefixes = ["TP9_", "AF7_", "AF8_", "TP10_"]
    eeg_features = [c for c in df.columns if any(c.startswith(p) for p in muse_prefixes)]
    return eeg_features


os.makedirs("models", exist_ok=True)

FEATURES_PATH = Path("dreamer_models/datasets/eego_features.csv")  # main training table
AROUSAL_MODEL_PATH = "models/arousal_lstm_muse.keras"
VALENCE_MODEL_PATH = "models/valence_lstm_muse.keras"

features_table = pd.read_csv(FEATURES_PATH)
print("eego_features shape:", features_table.shape)

# Sort to ensure consistent group ordering
sort_cols = ["user_id", "session_id", "affect_minute"]
features_table = features_table.sort_values(sort_cols).reset_index(drop=True)

# Rename EpocX channel prefixes -> Muse prefixes (so model expects Muse names)
features_table = rename_epocx_to_muse_columns(features_table)

features = select_eego_features(features_table)
print("Selected feature columns (Muse-style):")
print(features)
print("Number of feature columns:", len(features))

if os.path.exists(AROUSAL_MODEL_PATH) and os.path.exists(VALENCE_MODEL_PATH):
    print("Loading saved LSTM models from disk...")
    arousal_model = load_model(AROUSAL_MODEL_PATH)
    valence_model = load_model(VALENCE_MODEL_PATH)

else:
    DATA_PATH = Path("datasets/EEGo_labeled.csv")  # only needed if you want raw

    AROUSAL = "affect_arousal"
    VALENCE = "affect_valence"
    THRESH = 2.5

    LR = 1e-4
    EPOCHS = 10
    UNITS = 256
    BATCH_SIZE = 128
    PATIENCE = 10
    BIDIRECTIONAL = True

    FIXED_T = 2500

    def balance_binary_sequences(
        X: np.ndarray, y: np.ndarray, seed: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Downsample the majority class at the *sequence* level
        so that classes 0 and 1 are balanced.
        """
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

    ##### AROUSAL #####
    X_seq, arousal_seq = build_eego_lstm_sequences(
        features_table,
        feature_cols=features,
        target_col=AROUSAL,
        thresh=THRESH,
        fixed_T=FIXED_T,
    )
    print("X_seq shape:", X_seq.shape)
    print("arousal_seq shape:", arousal_seq.shape)
    print("Class counts:", np.bincount(arousal_seq.astype(int)))

    arousal_model, _, _ = train_lstm(
        X_seq,
        None,
        arousal_seq,
        None,
        lr=LR,
        epochs=EPOCHS,
        units=UNITS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        bidirectional=BIDIRECTIONAL,
    )

    arousal_model.save(AROUSAL_MODEL_PATH)
    print(f"Saved arousal model to {AROUSAL_MODEL_PATH}")
    ##### END AROUSAL #######

    ##### VALENCE #####
    X_seq, valence_seq = build_eego_lstm_sequences(
        features_table,
        feature_cols=features,
        target_col=VALENCE,
        thresh=THRESH,
        fixed_T=FIXED_T,
    )
    print("X_seq shape:", X_seq.shape)
    print("valence_seq shape:", valence_seq.shape)
    print("Class counts:", np.bincount(valence_seq.astype(int)))

    valence_model, _, _ = train_lstm(
        X_seq,
        None,
        valence_seq,
        None,
        lr=LR,
        epochs=EPOCHS,
        units=UNITS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        bidirectional=BIDIRECTIONAL,
    )

    valence_model.save(VALENCE_MODEL_PATH)
    print(f"Saved arousal model to {VALENCE_MODEL_PATH}")
    ##### END VALENCE #######