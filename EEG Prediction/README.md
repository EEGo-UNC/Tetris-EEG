# DEAP Emotion Recognition

This repository contains code to reproduce emotion recognition experiments on the **DEAP** dataset using **bandpower features**. It also includes code to run various models to compare results across multiple settings. 


## Deep Dataset Download

We use DEAP dataset from Kaggle. This by default will download it to the ```.cache``` folder. So the code modes it to the current directory. 

To download the dataset, run  - ```python3 deap_dataset.py```

---

## Experiments

### DEAP Baseline Classifier

This script implements **baseline emotion classification** experiments on the [DEAP dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/), which contains EEG and peripheral physiological recordings for emotion recognition.

The script:
- Extracts **EEG features** (log-bandpower in theta, alpha, beta, gamma bands)  
- Extracts **peripheral features** (GSR, Respiration, BVP, Skin Temperature statistics)  
- Binarizes DEAP labels (**valence, arousal, dominance, liking**) into *High* / *Low*  
- Trains classifiers using **SVMs with linear kernels**  
- Evaluates performance in two settings:
  - **Subject-dependent:** 10-fold CV within each subject
  - **Cross-subject (LOSO):** Leave-One-Subject-Out with EEG-only, peripherals-only, and EEG+peripherals fusion

Outputs include **Accuracy, Balanced Accuracy, and Macro F1-Score**.

Run the script - `python deap_baseline.py`


### DEAP Baseline Python Script

This notebook is complementary to `deap_baseline.py`, but structured for interactive experimentation.

This Jupyter notebook conducts **baseline experiments** on the **DEAP dataset** for emotion recognition.  
It focuses on:
- **Feature extraction** from EEG and peripheral signals
- **Classical ML models** (Linear SVM, calibrated classifiers)
- **Cross-validation evaluation** (subject-dependent and LOSO)

---
### MIL (Multiple-Instance Learning) for DEAP 

This Jupyter notebook implements **Multiple-Instance Learning (MIL)** with attention for emotion recognition on **EEG windowed features** from the **DEAP** dataset. It treats each **trial** as a *bag* of short-time **windows** (instances), learns attention weights over windows, and predicts **valence/arousal** (supports multi-output).

Run the script - It is a notebook. Run on Jupyter. 

---

### MLP for Emotion Classification (DEAP)

This Jupyter notebook implements **Multi-Layer Perceptrons (MLPs)** for classifying emotional states using the **DEAP dataset**.

It explores:
- **Single-output models**: predict either binary *valence* or *arousal* separately.
- **Multi-output models**: jointly predict binary *valence* and *arousal*.
- **Categorical outputs (1–9 scale)**: instead of binary classification, emotions are classified into one of 9 categories.

---

### ANN, SVM, RF, and KNN for Emotion Classification (DEAP)

This Jupyter notebook compares **shallow and traditional machine learning models** for classifying emotions using the **DEAP dataset**.  
It implements and evaluates:
- **ANN (Artificial Neural Network)** — a simple PyTorch feedforward model
- **SVM (Support Vector Machine)**
- **RF (Random Forest)**
- **KNN (K-Nearest Neighbors)**

It also explores **feature-level fusion** across EEG channels and reports **cross-validation results**.

---

### LSTM + AutoKeras (NOT RUN FULLY YET. LOSS IS NOT DIVERGING. NEEDS MODIFICATION)

This Jupyter notebook handles **data preprocessing** and **model prototyping** for the **DEAP dataset**.
It includes:
- Preparing EEG/label data from DEAP `.dat` files
- Splitting into train/test sets
- Building baseline models using LSTM and Autokeras
- **Models**
  - **LSTM**:
    - Implemented with TensorFlow/Keras (`layers.LSTM`, `Dense`, `Dropout`).
    - Configurable epochs, batch size, optimizer, and callbacks.
  - **AutoKeras**:
    - Automatically searches for a suitable neural architecture given the data.
    - Reduces manual hyperparameter tuning.

---

### Feature Importance Analysis

This notebook performs **comprehensive feature importance analysis** to identify which EEG features (channel-band combinations) have the most impact on predicting valence and arousal.

**Methods Used:**
1. **Correlation Analysis** — Pearson & Spearman correlation with targets
2. **Statistical Tests** — ANOVA F-test and Mutual Information
3. **Random Forest** — Tree-based feature importance scores
4. **Permutation Importance** — Model-agnostic importance by shuffling features
5. **Linear SVM Coefficients** — Weight analysis from linear models

**Outputs:**
- Aggregated feature rankings (averaging across all 5 methods)
- Frequency band analysis (alpha/beta/gamma importance)
- Brain region analysis (which channels contribute most)
- Visualizations (bar charts for top features)
- CSV files: `datasets/DEAP/filtered/feature_importance_valence.csv` and `datasets/DEAP/filtered/feature_importance_arousal.csv`
- PNGs: `datasets/DEAP/stats/*.png` (correlation, random forest, aggregated, band, channel)

**Configuration:**
Adjust the number of features to analyze by modifying these parameters at the top of the notebook:
```python
TOP_N_FEATURES = 30      # Number of features shown in tables
TOP_N_VISUALIZE = 20     # Number of features in plots
TOP_N_AGGREGATE = 50     # Features in final aggregated ranking
```

Run the notebook: `notebooks/Feature_Importance_Analysis.ipynb`

---

### Create Filtered Dataset

This notebook generates **filtered datasets** containing only the most important features identified by the Feature Importance Analysis.

**What it Creates:**
1. **Valence-Optimized Dataset** — Top N features for valence prediction
2. **Arousal-Optimized Dataset** — Top N features for arousal prediction
3. **Combined Dataset** — Top N features for both valence and arousal

**Output Formats:**
- **NumPy arrays** (.npy) for fast loading
- **CSV files** with feature names and labels
- **Feature lists** (.txt) documenting which features were selected
- **README.md** with complete metadata

**Configuration:**
Adjust the number of features to include:
```python
TOP_N_VALENCE = 30   # Features for valence-optimized dataset
TOP_N_AROUSAL = 30   # Features for arousal-optimized dataset
TOP_N_COMBINED = 40  # Features for combined dataset
```

**Usage Example:**
```python
import numpy as np

# Load filtered dataset
X = np.load('datasets/DEAP/filtered_features/X_valence_top30.npy')
y = np.load('datasets/DEAP/filtered_features/y_valence_binary.npy')

# Train model with reduced feature set (30 instead of 96)
```

**Benefits:**
- Faster training time
- Reduced risk of overfitting
- Better model interpretability
- Focus on most predictive features

Run the notebook: `notebooks/Create_Filtered_Dataset.ipynb`

---

## DREAMER Dataset Analysis

The DREAMER dataset analysis notebooks provide the same comprehensive feature importance and dataset filtering capabilities for the DREAMER EEG emotion recognition dataset.

### DREAMER Feature Importance Analysis

This notebook analyzes feature importance for the **DREAMER dataset**, which contains:
- **14 EEG channels**: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- **PSD + Shannon entropy features** from `features_table_psd_shannon.csv`
- Optional frequency decompositions (bandpower, mobility, complexity, wenergy, IMF) when using merged tables

**Key Features:**
- Supports both merged (`features_table.csv` + `features_table_imf.csv`) and PSD/Shannon-only (`features_table_psd_shannon.csv`) workflows
- Uses same 5 methods as DEAP analysis (Correlation, F-test, Random Forest, Permutation, SVM)
- Generates rankings for hundreds of features per configuration
- Configurable binarization threshold for 1-5 rating scale
- Outputs CSV files (`datasets/Dreamer/filtered/dreamer_feature_importance_*.csv`) and visualizations (`datasets/Dreamer/stats/*.png`)

Run the notebook: `notebooks/DREAMER_Feature_Importance_Analysis.ipynb`

### DREAMER Create Filtered Dataset

This notebook generates filtered DREAMER datasets with only the top important features.

**Configuration:**
Adjust these parameters at the top of the notebook:
```python
TOP_N_VALENCE = 30       # Features for valence-optimized dataset
TOP_N_AROUSAL = 30       # Features for arousal-optimized dataset
TOP_N_COMBINED = 40      # Features for combined dataset
BINARIZE_THRESHOLD = 3.0 # Threshold for binarizing (1-5 scale)
```

**Outputs:**
- NumPy arrays, CSV files, and feature lists
- Saved to `datasets/Dreamer/filtered_features/`
- README.md with complete metadata

Run the notebook: `notebooks/DREAMER_Create_Filtered_Dataset.ipynb`

---

## Directory Structure

```
EEG Prediction/
│
├── datasets/
│   ├── DEAP/
│   │   ├── deap-dataset/               # DEAP dataset files (not included, download separately)
│   │   │   ├── audio_stimuli_MIDI/
│   │   │   ├── data_preprocessed_python/  # s01.dat ... s32.dat
│   │   │   ├── extracted_features/     # Feature-level data (CSV per channel)
│   │   │   ├── Metadata/               # participant_ratings.xls
│   │   │   └── EDA_DEAP.ipynb
│   │   │
│   │   ├── filtered/                   # Feature-importance rankings (CSV)
│   │   │   ├── feature_importance_valence.csv
│   │   │   └── feature_importance_arousal.csv
│   │   │
│   │   ├── stats/                      # Plots generated by feature importance notebook
│   │   │   ├── correlation_analysis.png
│   │   │   ├── random_forest_importance.png
│   │   │   ├── top_features_aggregated.png
│   │   │   ├── band_importance.png
│   │   │   └── channel_importance.png
│   │   │
│   │   └── filtered_features/          # DEAP filtered datasets (generated)
│   │       ├── X_valence_top30.npy
│   │       ├── X_arousal_top30.npy
│   │       ├── X_combined_top40.npy
│   │       ├── y_valence_binary.npy
│   │       ├── y_arousal_binary.npy
│   │       ├── dataset_*.csv
│   │       └── README.md
│   │
│   └── Dreamer/
│       ├── features_table.csv          # Main DREAMER features (bandpower, activity, etc.)
│       ├── features_table_imf.csv      # IMF features (IMF energy, IMF entropy)
│       ├── features_table_psd_shannon.csv
│       │
│       ├── filtered/                   # DREAMER feature-importance rankings (CSV)
│       │   ├── dreamer_feature_importance_valence.csv
│       │   └── dreamer_feature_importance_arousal.csv
│       │
│       ├── stats/                      # DREAMER feature-importance plots
│       │   ├── dreamer_correlation_analysis.png
│       │   ├── dreamer_random_forest_importance.png
│       │   └── dreamer_top_features_aggregated.png
│       │
│       └── filtered_features/          # DREAMER filtered datasets (generated)
│           ├── X_valence_top30.npy
│           ├── X_arousal_top30.npy
│           ├── X_combined_top40.npy
│           ├── y_valence_binary.npy
│           ├── y_arousal_binary.npy
│           ├── dataset_*.csv
│           └── README.md
│
├── notebooks/                          # Jupyter notebooks for experiments
│   ├── ANN_SVM_RF_KNN.ipynb            # Classical ML models + ANN
│   ├── deap_baseline.ipynb             # Baseline experiments with SVMs
│   ├── LSTM_Autokeras.ipynb            # Preprocessing + LSTM + AutoKeras models
│   ├── MIL.ipynb                       # Multiple-Instance Learning with attention
│   ├── MLP.ipynb                       # Multi-Layer Perceptrons (single & multi-output)
│   │
│   ├── Feature_Importance_Analysis.ipynb        # DEAP feature importance
│   ├── Create_Filtered_Dataset.ipynb            # DEAP filtered datasets
│   ├── DREAMER_Feature_Importance_Analysis.ipynb  # DREAMER feature importance
│   └── DREAMER_Create_Filtered_Dataset.ipynb      # DREAMER filtered datasets
│
├── scripts/                            # Standalone Python scripts
│   └── deap_baseline.py                # Baseline feature extraction + classification
│
├── .gitignore                          # Ignored files (datasets, checkpoints, etc.)
├── deap_dataset.py                     # Utility for dataset loading/preprocessing
├── README.md                           # Project overview & instructions
└── requirements.txt                    # Dependencies
```

## Contibute

### 1. Setup Environment
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/EEG-Prediction.git
cd EEG-Prediction
pip install -r requirements.txt
```

---

### 2. Adding New Experiments
- **New notebooks** → put inside `notebooks/`  
  Example: `notebooks/Transformer_EEG.ipynb`
- **Reusable scripts** → put inside `scripts/` if it’s meant for multiple notebooks  
  Example: `scripts/feature_extraction.py`
- **Dataset paths** → always use relative paths (e.g., `./datasets/DEAP/...`) so code runs on any machine.

---

### 3. Naming Conventions
- Python files: `snake_case.py`
- Notebooks: `ModelName_Description.ipynb` (e.g., `CNN_EmotionNet.ipynb`)
- Functions/classes: follow PEP8 standards.

---

### 4. Updating Requirements
If you add a new dependency:
1. Install it locally
   ```bash
   pip install <package>
   ```
2. Pin the version and update requirements
   ```bash
   pip freeze | grep <package> >> requirements.txt
   ```

---


