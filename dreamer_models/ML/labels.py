import pandas as pd
import numpy as np
from .utils import read_table

BASIS_DICT = {
    0: {"valence": 4, "arousal": 1},  # calmness
    10: {"valence": 4, "arousal": 1},  # calmness
    1: {"valence": 3, "arousal": 5},  # surprise
    17: {"valence": 3, "arousal": 5},  # surprise
    2: {"valence": 5, "arousal": 4},  # amusement
    11: {"valence": 5, "arousal": 4},  # amusement
    3: {"valence": 1, "arousal": 5},  # fear
    14: {"valence": 1, "arousal": 5},  # fear
    4: {"valence": 5, "arousal": 5},  # excitement
    15: {"valence": 5, "arousal": 5},  # excitement
    5: {"valence": 1, "arousal": 3},  # disgust
    9: {"valence": 1, "arousal": 3},  # disgust
    6: {"valence": 5, "arousal": 3},  # happiness
    12: {"valence": 5, "arousal": 3},  # happiness
    7: {"valence": 1, "arousal": 4},  # anger
    13: {"valence": 1, "arousal": 4},  # anger
    8: {"valence": 1, "arousal": 2},  # sadness
    16: {"valence": 1, "arousal": 2},  # sadness
}


def hybrid_video_label_mean(
    df_main: pd.DataFrame,
    target: str = "arousal",
    video_col: str = "video_index",
    patient_col: str = "patient_index",
    w_video_mean: float = 0.5,  #
    w_basis: float = 0.5,  
    w_user: float = 0.5,  
) -> pd.DataFrame:
    """
    For each row (user, video), build a hybrid label that is a weighted
    average of:
        - the mean rating for that video across all users,
        - the BASIS_DICT label for that video,
        - the user's own rating for that video.

    Returns a copy of df_main with `target` replaced by the hybrid value.
    """
    
    video_means = (
        df_main.groupby(video_col, dropna=False)[target].mean().rename("video_mean")
    )

    
    basis_series = pd.Series(
        {
            vid: lab[target] if isinstance(lab, dict) else lab
            for vid, lab in BASIS_DICT.items()
        },
        name="basis_label",
        dtype="float",
    )

    present_videos = pd.Index(df_main[video_col].unique())
    video_means = video_means.reindex(present_videos)
    basis_series = basis_series.reindex(present_videos)

    vm = df_main[video_col].map(video_means)  
    bs = df_main[video_col].map(basis_series)  
    ur = df_main[target].astype(float)  

    
    denom = w_video_mean + w_basis + w_user
    combined = (w_video_mean * vm + w_basis * bs + w_user * ur) / denom
    combined.name = f"{target}_hybrid_mean"

    out = df_main.copy()
    out[target] = combined

    return out


def mean_labels(
    df_main: pd.DataFrame,
    target: str = "arousal",
) -> pd.DataFrame:
    per_pv = (
        df_main.groupby(["patient_index", "video_index"], dropna=False)[target]
        .mean()
        .reset_index()
    )

    video_means = (
        per_pv.groupby("video_index", dropna=False)[target]
        .mean()
        .rename(f"{target}_video_mean")
    )

    mapped = df_main["video_index"].map(video_means)

    out = df_main.copy()
    out[target] = mapped

    return out


def relabel_target_from_video_map(
    df_main: pd.DataFrame,
    target: str = "arousal",
    strict: bool = True,
):
    flat_map = {}
    for vid, val in BASIS_DICT.items():
        flat_map[vid] = val[target]

    out = df_main.copy()
    mapped = out["video_index"].map(flat_map)

    if strict and mapped.isna().any():
        missing_vids = sorted(out.loc[mapped.isna(), "video_index"].unique().tolist())
        raise KeyError(f"No mapping for video_index values: {missing_vids}")

    out[target] = mapped.where(~mapped.isna(), out.get(target, out.get(target)))

    return out


def build_video_rating_tables(
    df: pd.DataFrame,
    user_col: str = "patient_index",
    video_col: str = "video_index",
    rating_cols: tuple[str] = ("arousal", "valence"),
    agg: str = "mean",
):
    ratings = {}
    if agg == "list":
        aggregator = lambda s: list(s)
    else:
        aggregator = agg  # e.g., 'mean', 'median', np.mean, etc.

    for r in rating_cols:
        mat = (
            df.groupby([video_col, user_col])[r]
            .agg(aggregator)
            .unstack(user_col)
            .sort_index(axis=0)  # sort videos
            .sort_index(axis=1)  # sort users
        )
        ratings[r] = mat

    return ratings


def find_z_outliers(
    rating_tables: dict[str, pd.DataFrame],
    threshold: float = 1.0,
    min_count: int = 3,
):
    outliers = {}

    for rating_name, df in rating_tables.items():

        def row_z(row: pd.Series) -> pd.Series:
            s = row.dropna()
            if s.size < min_count:
                return pd.Series(index=row.index, dtype=float)
            mu = s.mean()
            sd = s.std(ddof=1)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(np.nan, index=row.index)
            return (row - mu) / sd

        Z = df.apply(row_z, axis=1)
        mask = Z.abs() > threshold

        per_video = {}
        for vid in df.index:
            cols = mask.columns[mask.loc[vid] == True]
            if len(cols):
                entries = []
                for user in cols:
                    v = df.loc[vid, user]
                    z = Z.loc[vid, user]
                    entries.append(user)
                per_video[int(vid)] = entries
        outliers[rating_name] = per_video

    return outliers


def remove_outlier_videos(
    df: pd.DataFrame,
    target: str = "arousal",
    range_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Remove all rows for videos where the across-user response range > range_threshold.
    """
    rating_tables = build_video_rating_tables(df)
    table = rating_tables[target]
    resp_range = table.max(axis=1) - table.min(axis=1)

    remove_videos = resp_range.index[(resp_range > range_threshold)].tolist()
    print(remove_videos)

    filtered = df.loc[~df["video_index"].isin(remove_videos)].reset_index(drop=True)
    return filtered


if __name__ == "__main__":
    df = read_table("datasets/features_table.csv").reset_index(drop=True)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    print(build_video_rating_tables(df)["arousal"])
    remove_outlier_videos(df)
