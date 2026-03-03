import pandas as pd
import numpy as np
from .utils import read_table
from sklearn.model_selection import train_test_split
from .labels import *


def omit_eego_sesh(filename="datasets/EEGo_labeled.csv", omit_session_ids: list[str] = None):
    df = pd.read_csv(filename)

    if omit_session_ids is None:
        return df.reset_index(drop=True)

    if isinstance(omit_session_ids, str):
        omit_set = {omit_session_ids}
    else:
        omit_set = set(omit_session_ids)

    mask = ~df["session_id"].isin(omit_set)
    df_omit = df.loc[mask].reset_index(drop=True)

    return df_omit


def random_train_test_split(
    test_size: float = 0.1,
    target: str = "arousal",
    shuffle_random_state: int | None = None,
):
    df_main = read_table("datasets/features_table.csv").reset_index(drop=True)
    df = df_main.drop(columns=["Unnamed: 0"], errors="ignore").reset_index(drop=True)

    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in dataframe columns.")

    y = df[target].astype(float)

    blacklist = {
        "patient_index",
        "video_index",
        "arousal",
        "valence",
        "Unnamed: 0",
        target,
    }

    X = df.drop(columns=[c for c in blacklist if c in df.columns], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=shuffle_random_state,
        stratify=None,
    )

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def omit_patient_video(  # Leave-N-trials-out for one patient, with optional manual holds
    target: str = "arousal",
    random_state: int | None = None,
    trials: int = 3,
    exclude_users: list[int] | set[int] | None = None,
    selected_user: int | None = None,
    holdout_videos: (
        list[int] | None
    ) = None,
    filename: str = "datasets/features_table.csv",
    relabel = False,
):
    if trials < 1:
        raise ValueError("`trials` must be >= 1.")
    
    df_main = read_table(filename).reset_index(drop=True)
    df_main = df_main.drop(columns=["Unnamed: 0"], errors="ignore")
    if relabel is True:
        df_main = relabel_target_from_video_map(df_main)


    if exclude_users:
        df_main = df_main[~df_main["patient_index"].isin(exclude_users)].reset_index(
            drop=True
        )
    vids_per_patient = df_main.groupby("patient_index")["video_index"].nunique()
    eligible_patients = vids_per_patient.index.to_numpy()
    if eligible_patients.size == 0:
        raise ValueError("No patients left after exclusions.")

    rng = np.random.default_rng(random_state)

    if selected_user is not None:
        if selected_user not in vids_per_patient.index:
            raise ValueError(
                f"Selected user {selected_user} not present after exclusions."
            )
        chosen_patient = int(selected_user)
    else:
        if holdout_videos is None:
            eligible_with_trials = vids_per_patient[
                vids_per_patient >= trials
            ].index.to_numpy()
            if eligible_with_trials.size == 0:
                raise ValueError(
                    "No eligible patients with enough videos for the requested `trials`."
                )
            chosen_patient = int(rng.choice(eligible_with_trials))
        else:
            chosen_patient = int(rng.choice(eligible_patients))

    patient_mask = df_main["patient_index"] == chosen_patient
    patient_videos = pd.unique(df_main.loc[patient_mask, "video_index"])

    if holdout_videos is not None:
        held_videos = np.array(sorted(set(map(int, holdout_videos))), dtype=int)
        missing = [v for v in held_videos if v not in set(patient_videos)]
        if missing:
            raise ValueError(
                f"Holdout videos not found for user {chosen_patient}: {missing}"
            )
        trials = len(held_videos)
    else:
        if len(patient_videos) < trials:
            raise ValueError(
                f"User {chosen_patient} has only {len(patient_videos)} videos, but trials={trials}."
            )
        held_videos = rng.choice(patient_videos, size=trials, replace=False)

    held_out_trials = [(chosen_patient, int(v)) for v in held_videos]

    # X = df_main.drop(
    #     columns=["arousal", "valence"], errors="ignore"
    # )
    X = df_main
    y = df_main[target]

    trial_id = pd.Series(
        list(map(tuple, df_main[["patient_index", "video_index"]].to_numpy())),
        index=df_main.index,
    )
    if not (len(X) == len(y) == len(trial_id)):
        raise ValueError("X, y, and trial_id must have the same number of rows.")

    test_mask = trial_id.isin(set(held_out_trials)).to_numpy()
    test_idx = np.nonzero(test_mask)[0]
    train_idx = np.nonzero(~test_mask)[0]

    print(
        f"Held-out patient: {chosen_patient} | Held-out (patient, video) trials:",
        sorted(held_out_trials, key=lambda t: (t[0], t[1])),
        "| Excluded users:",
        sorted(exclude_users) if exclude_users else [],
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def omit_patient(  # PERFORMS LOSO (Leave-one--out)
    target: str = "arousal", held_out_patient: int = None
):
    df_main = read_table("datasets/features_table.csv").reset_index(drop=True)
    df_main = df_main.drop(columns=["Unnamed: 0"], errors="ignore")
    # df_main = hybrid_video_label_mean(df_main, target)
    # df_main = mean_labels(df_main, target)
    # df_main = relabel_target_from_video_map(df_main, target)
    # participants_to_drop = [0, 1, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]
    # df_main = df_main[~df_main["patient_index"].isin(participants_to_drop)].reset_index(
    #     drop=True
    # )

    # X = df_main.drop(
    #     columns=["arousal", "valence"],
    #     errors="ignore",
    # )
    X = df_main.drop(
        columns=["arousal", "valence"],
        errors="ignore",
    )
    y = df_main[target]

    patient_index = df_main["patient_index"]
    patient_index = pd.Series(patient_index).reset_index(drop=True)

    if held_out_patient is None:
        rng = np.random.default_rng()
        unique_patients = pd.unique(patient_index)
        held_out_patient = rng.choice(unique_patients)

    test_mask = (patient_index == held_out_patient).to_numpy()
    test_idx = np.nonzero(test_mask)[0]
    train_idx = np.nonzero(~test_mask)[0]
    print(held_out_patient)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def single_user_split(
    target: str,
    selected_user: int | None = None,
    holdout_videos: list[int] | None = None,
    k_holdouts: int = 0,
    random_state=None,
    filename="datasets/feature_table.csv"
):
    """Splits one trial-user combination and returns only one user's data in the train set."""
    df = read_table(filename).reset_index(drop=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    # df = remove_outlier_videos(df, target)
    df = relabel_target_from_video_map(df)

    rng = np.random.default_rng(random_state)
    if selected_user is None:
        users = df["patient_index"].unique()
        selected_user = int(rng.choice(users))

    videos = df["video_index"].unique()
    if holdout_videos is None:
        holdouts = rng.choice(videos, size=k_holdouts, replace=False)
    else:
        holdouts = np.array(sorted(set(holdout_videos)), dtype=int)

    # print(selected_user, holdouts)
    #  X = df.drop(
    #     columns=["arousal", "valence"], errors="ignore"
    # )
    X = df
    y = df[target]

    mask = df["patient_index"] == selected_user
    trial_mask = mask & (df["video_index"].isin(holdouts))

    X_train = X.loc[mask & ~trial_mask].reset_index(drop=True)
    y_train = y.loc[mask & ~trial_mask].reset_index(drop=True)

    X_test = X.loc[trial_mask].reset_index(drop=True)
    y_test = y.loc[trial_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test
