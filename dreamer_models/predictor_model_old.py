import numpy as np
import os

from .ML.model_training import train_lstm, build_lstm_sequences
from .ML.utils import filter_features, read_table
from tensorflow.keras.models import load_model


AROUSAL_MODEL_PATH = "models/arousal_lstm.keras"
VALENCE_MODEL_PATH = "models/valence_lstm.keras"
X = read_table("datasets/experiment_features_table.csv")

features = filter_features(
    X.columns,
    remove_bands=["gamma", "delta"],
)
features.remove("Unnamed: 0")
features.remove("patient_index")
features.remove("video_index")
features.remove("arousal")
features.remove("valence")


os.makedirs("models", exist_ok=True)

if os.path.exists(AROUSAL_MODEL_PATH) and os.path.exists(VALENCE_MODEL_PATH):
    print("Loading saved LSTM models from disk...")
    arousal_model = load_model(AROUSAL_MODEL_PATH)
    valence_model = load_model(VALENCE_MODEL_PATH)

else:
    print("Saved models not found. Training from scratch...")
    features = [c for c in features if c in X.columns]
    X_train_seq, arousal_train_seq = build_lstm_sequences(
        X, features, target_col="arousal", thresh=3.8, fixed_T=76
    )
    print("arousal_train counts:", np.bincount(arousal_train_seq.astype(int)))

    arousal_model, X_test_eval, y_test_eval = train_lstm(
        X_train_seq,
        None,
        arousal_train_seq,
        None,
        lr=0.0001,
        epochs=100,
        units=512,
        batch_size=64,
        dropout=0.4,
        recurrent_dropout=0.2,
        bidirectional=True,
    )
    arousal_model.save(AROUSAL_MODEL_PATH)
    print(f"Saved arousal model to {AROUSAL_MODEL_PATH}")

    X_train_seq, valence_train_seq = build_lstm_sequences(
        X, features, target_col="valence", thresh=3.8, fixed_T=76
    )
    print("valence_train counts:", np.bincount(valence_train_seq.astype(int)))

    valence_model, X_test_eval, y_test_eval = train_lstm(
        X_train_seq,
        None,
        valence_train_seq,
        None,
        lr=0.0001,
        epochs=100,
        units=512,
        batch_size=64,
        dropout=0.4,
        recurrent_dropout=0.2,
        bidirectional=True,
    )
    valence_model.save(VALENCE_MODEL_PATH)
    print(f"Saved valence model to {VALENCE_MODEL_PATH}")
