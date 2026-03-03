# DREAMER Models

Python 3.10+, NumPy/Pandas, SciPy, scikit-learn, MNE (for ICA), PyTorch (RF variant), Matplotlib.

```
dreamer_models/
├─ best_tests/
│   ├─ best_all_user_LOSO.txt
│   ├─ best_LOO.txt
│   ├─ best_result_single_user_split.txt
│   └─ single_user_LOO_no_drops.txt
│
├─ datasets/                     # Raw/processed DREAMER CSVs (local only)
│
├─ dreamer_extraction/
│   └─ start_dreamer.ipynb       # DREAMER parsing/setup notebook
│
├─ ML/
│   ├─ __init__.py
│   ├─ features.ipynb            # Feature generation notebook
│   ├─ ICA.py                    # EEG-only ICA helpers for artifact removal
│   ├─ labels.py                 # Label parsing and relabel helpers
│   ├─ model_testing.ipynb       # Result validation / confusion matrices
│   ├─ model_training.py         # Training utilities, splits, metrics, saves
│   ├─ splits.py                 # Group splits for DREAMER (LOO, omit video, etc.)
│   ├─ STSNet.py                 # Experimental STSNet architecture
│   ├─ utils.py                  # Filtering, loading, subsets, asymmetry features
│   └─ __pycache__/              # Auto-generated cache
│
├─ experiment_models.ipynb       # LOSO PSD only + fine-tuning
├─ LOSO_models.ipynb              # Leave-One-User-Out  + fine-tuning
├─ predictor_model.py            # Model used for real-time/in-game predictions
├─ README.md
└─ single_user_models.ipynb      # Per-user or subject-specific modeling

```

## predictor_model.py
This script trains a model that can be imported by other parts of this project, namely, in the `experiment` portion. The script trains, saves, and gets lstm model as `.keras` file.
Then, `arousal_model` and `valence_model` can be imported to do in-game speed change predictions over the real-time collected EEG data.


## model_training.py
- random_train_test_split(...): Loads features_table.csv, builds X,y for target, drops metadata, and returns a standard randomized train/test split.
- omit_patient_video(...): Loads main features, groups by patient_index_video and does a GroupShuffleSplit so entire patient–video pairs are held out together; returns grouped train/test splits.
- train_random_forest(X_train, X_test, y_train, y_test, ...): Fits a RandomForestRegressor (pre-set to 300 trees, full depth) and returns the trained model with the untouched test sets.
- train_knn(X_train, X_test, y_train, y_test, ...): Fits a distance-weighted KNeighborsRegressor (default k=1, p=1) and returns the model with the test sets.
train_svr(X_train, X_test, y_train, y_test): Fits an RBF-kernel SVR (C=1.0, ε=0.1) and returns the model with the test sets.


## ML/utils.py
- read_table(filename="datasets/features_table.csv"): Resolves the repo path and loads a CSV into a pandas.DataFrame, raising FileNotFoundError if missing.
- generate_all_subsets(columns, sensor_families=None, feature_tokens=None, freq_tokens=None, min_size=6): Parses columns into (sensor family, feature token, frequency band) and returns unique feature subsets formed by all combinations of present families/features/bands, filtered to at least min_size.
- HOMOLOGOUS_PAIRS: Left–right channel pairs for Emotiv/DREAMER (e.g., (AF3, AF4), (F3, F4)) used to compute hemispheric asymmetries.
- AS_BANDS: Allowed band names for asymmetry computation: {delta, theta, alpha, beta, gamma}.
- compute_asymmetry_from_psd(psd, pairs=HOMOLOGOUS_PAIRS, eps=1e-12, add_log=True, prefix_da="da", prefix_ra="ra"): From PSD features like CH_band, computes directional asymmetry da (log- or linear difference) and rational asymmetry ra = (PR−PL)/(PR+PL+eps) for each pair×band, returning a new DataFrame of asymmetry columns.

