import pandas as pd


def _get_delay(threshold_in_minutes, min_diff: float, threshold: float = None) -> int:
    """
    Return 1 if min_diff is greater than the threshold (default from config), else 0.
    """
    if threshold is None:
        threshold = threshold_in_minutes
    return 1 if min_diff > threshold else 0


def _get_period_day(fecha: pd.Timestamp) -> str:
    """
    Return the period of the day based on 'Fecha-I'.

    - 'morning': 5:00 <= hour < 12:00
    - 'afternoon': 12:00 <= hour < 19:00
    - 'night': otherwise
    """
    hour = fecha.hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 19:
        return "afternoon"
    else:
        return "night"


def _is_high_season(fecha: pd.Timestamp) -> int:
    """
    Return 1 if 'Fecha-I' is in high season, else 0.
    High season is defined as:
    - Dec 15 to Mar 3,
    - Jul 15 to Jul 31,
    - Sep 11 to Sep 30.
    """
    month = fecha.month
    day = fecha.day
    cond_dec_mar = (
        (month == 12 and day >= 15) or (month in [1, 2]) or (month == 3 and day <= 3)
    )
    cond_jul = month == 7 and 15 <= day <= 31
    cond_sep = month == 9 and 11 <= day <= 30

    return 1 if (cond_dec_mar or cond_jul or cond_sep) else 0


def compute_vectorized_features(threshold_in_minutes, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features using vectorized operations:
    - min_diff: Time difference in minutes.
    - delay: 1 if min_diff > threshold_in_minutes, else 0.
    - period_day: Derived from the hour of 'Fecha-I'.
    - high_season: 1 if 'Fecha-I' is in high season, else 0.

    Args:
        threshold_in_minutes: Threshold in minutes to method get delay
        df (pd.DataFrame): DataFrame with valid datetime columns 'Fecha-I' and 'Fecha-O'.

    Returns:
        pd.DataFrame: DataFrame with additional columns for derived features.
    """
    # Calculate min_diff vectorized
    df["min_diff"] = (df["Fecha-O"] - df["Fecha-I"]).dt.total_seconds() / 60.0
    # Compute delay vectorized using the imported helper function
    df["delay"] = (
        df["min_diff"].apply(lambda x: _get_delay(threshold_in_minutes, x))
    ).astype(int)
    # Compute period_day vectorized using the imported helper
    df["period_day"] = df["Fecha-I"].apply(_get_period_day)
    # Compute high_season vectorized using the imported helper
    df["high_season"] = df["Fecha-I"].apply(_is_high_season)
    return df