import os, pickle, warnings
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm
from sklearn.svm import SVC  # add this


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "datasets/DEAP/deap-dataset/data_preprocessed_python"  # folder containing s01.dat ... s32.dat
SUBJECTS = [f"s{idx:02d}.dat" for idx in range(1, 33)]

FS = 128
TRIAL_SEC = 60
BASELINE_SEC = 3
KEEP_SAMPLES = FS * TRIAL_SEC              # 7680
DROP_SAMPLES = FS * BASELINE_SEC           # 384 (drop from start)

EEG_CH = 32                                # first 32 channels are EEG
BANDS = [(4,8), (8,13), (13,30), (30,45)]  # theta, alpha, beta, gamma
LABELS = ["valence", "arousal", "dominance", "liking"]
TARGET = "valence"                         # change to "arousal"/"liking"/"dominance" as needed

# We are using 4 robust peripheral channels commonly present after EEG - GSR, Respiration, BVP, Skin Temp (indices relative to *peripheral* block)
PERIPH_KEEP = [0, 1, 2, 3]

def binarize(y_cont):
    """DEAP convention: rating > 5 => 1 (High), else 0 (Low)."""
    return (y_cont > 5.0).astype(np.int32)

def load_subject_dat(path):
    """
    Load DEAP preprocessed .dat (pickle).
    Returns:
      eeg:    (40, 32, 7680)
      periph: (40, P, 7680) or None
      labels: (40, 4)
    """
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")  # DEAP pickles were py2
    data = obj["data"]         # shape (40, 40, 8064): trials, channels, samples
    labels = obj["labels"]     # shape (40, 4): valence, arousal, dominance, liking (1..9)

    # split channels
    eeg = data[:, :EEG_CH, :]                  # (40, 32, 8064)
    periph = data[:, EEG_CH:, :]               # (40, 8, 8064)

    # keep only the 60s trial segment (drop first 3s baseline)
    eeg = eeg[:, :, DROP_SAMPLES:DROP_SAMPLES + KEEP_SAMPLES]          # (40, 32, 7680)
    periph = periph[:, :, DROP_SAMPLES:DROP_SAMPLES + KEEP_SAMPLES]    # (40, 8, 7680)

    # keep robust peripheral subset
    if periph.shape[1] >= 4:
        periph = periph[:, PERIPH_KEEP, :]
    else:
        periph = None

    return eeg.astype(np.float32), (periph.astype(np.float32) if periph is not None else None), labels.astype(np.float32)

def welch_logbp(x, fs=FS, nperseg=256, noverlap=128):
    f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    out = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f <= hi)
        bp = np.trapz(Pxx[m], f[m]) + 1e-12
        out.append(np.log(bp))
    return np.asarray(out, dtype=np.float32)

def eeg_features(trial_eeg):
    # trial_eeg: (32, 7680)
    feats = [welch_logbp(trial_eeg[ch]) for ch in range(trial_eeg.shape[0])]
    return np.concatenate(feats, axis=0)  # (32 * 4,)

def periph_features(trial_periph):
    # trial_periph: (P, 7680); simple trial-level stats
    def slope(x):
        t = np.arange(x.size, dtype=np.float32)
        A = np.vstack([t, np.ones_like(t)]).T
        m, _ = np.linalg.lstsq(A, x, rcond=None)[0]
        return np.float32(m)
    feats = []
    for ch in range(trial_periph.shape[0]):
        sig = trial_periph[ch]
        feats.extend([np.mean(sig), np.std(sig), slope(sig)])
    return np.asarray(feats, dtype=np.float32)  # (P*3,)

def build_features(eeg, periph):
    X_eeg = np.vstack([eeg_features(eeg[i]) for i in range(eeg.shape[0])])
    X_per = None
    if periph is not None:
        X_per = np.vstack([periph_features(periph[i]) for i in range(periph.shape[0])])
    return X_eeg, X_per

def metric_dict(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro"))
    }

def subject_dependent_cv(X, y, C=1.0):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = SVC(kernel="linear", C=C, probability=True, class_weight="balanced", random_state=42)
        clf.fit(Xtr, y[tr])
        yhat = clf.predict(Xte)
        scores.append(metric_dict(y[te], yhat))
    return {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}

def loso(mods, labels, C=1.0):
    """
    mods: dict modality -> list of per-subject X arrays
    labels: list of per-subject y arrays
    returns averaged metrics per modality and for fusion
    """
    subj_n = len(labels)
    res = {k: [] for k in mods.keys()}
    res["fusion"] = []

    for te in range(subj_n):
        # set up per-modality models on train subjects
        permod = {}
        for mod, Xs in mods.items():
            Xtr = np.vstack([Xs[i] for i in range(subj_n) if i != te])
            ytr = np.hstack([labels[i] for i in range(subj_n) if i != te])
            Xte = Xs[te]
            yte = labels[te]

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

            clf = SVC(kernel="linear", C=C, probability=True, class_weight="balanced", random_state=42)
            clf.fit(Xtr, ytr)
            yhat = clf.predict(Xte)
            proba = clf.predict_proba(Xte)
            permod[mod] = (yte, yhat, proba)

        # per-modality metrics
        for mod, (yte, yhat, proba) in permod.items():
            res[mod].append(metric_dict(yte, yhat))

        # simple late fusion: average calibrated probabilities
        probs = np.mean(np.stack([v[2] for v in permod.values()], axis=0), axis=0)
        yhat_f = np.argmax(probs, axis=1)
        yte0 = next(iter(permod.values()))[0]
        res["fusion"].append(metric_dict(yte0, yhat_f))

    return {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in res.items()}

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    all_eeg, all_per, all_y = [], [], []

    print("Loading .dat files and extracting features …")
    for fname in tqdm(SUBJECTS):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        eeg, periph, labels = load_subject_dat(fpath)
        Xeeg, Xper = build_features(eeg, periph)

        # choose target label and binarize
        t_idx = LABELS.index(TARGET)
        y = binarize(labels[:, t_idx])

        all_eeg.append(Xeeg)
        if Xper is not None:
            all_per.append(Xper)
        else:
            all_per.append(np.zeros((Xeeg.shape[0], 0), dtype=np.float32))
        all_y.append(y)

    # ---- Subject-dependent baseline (EEG only) ----
    sub_metrics = []
    for i in range(len(all_eeg)):
        sub_metrics.append(subject_dependent_cv(all_eeg[i], all_y[i], C=1.0))
    avg_sub = {k: float(np.mean([m[k] for m in sub_metrics])) for k in sub_metrics[0]}
    print("\n=== Subject-dependent (EEG) — 10-fold within subject ===")
    print(avg_sub)

    # ---- LOSO across subjects (EEG, peripherals, fusion) ----
    mods = {"eeg": all_eeg, "periph": all_per}
    loso_avg = loso(mods, all_y, C=1.0)
    print("\n=== LOSO Cross-Subject ===")
    for k, v in loso_avg.items():
        print(k, v)
