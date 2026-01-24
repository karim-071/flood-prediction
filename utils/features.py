import pandas as pd

def add_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Monthly columns (Jan–Dec)
    month_cols = df.columns[2:14]

    # Annual rainfall
    df["annual_rainfall"] = df[month_cols].sum(axis=1)

    # Monsoon rainfall (Jun–Sep)
    df["monsoon_rainfall"] = df.iloc[:, 7:11].sum(axis=1)

    # Pre-monsoon (Mar–May)
    df["pre_monsoon_rainfall"] = df.iloc[:, 4:7].sum(axis=1)

    # Post-monsoon (Oct–Dec)
    df["post_monsoon_rainfall"] = df.iloc[:, 11:14].sum(axis=1)

    return df

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Group by state for state-specific context
    df["monsoon_mean"] = df.groupby("SUBDIVISION")["monsoon_rainfall"].transform("mean")
    df["monsoon_std"] = df.groupby("SUBDIVISION")["monsoon_rainfall"].transform("std")

    # Z-score (anomaly detection)
    df["monsoon_anomaly"] = (
        df["monsoon_rainfall"] - df["monsoon_mean"]
    ) / df["monsoon_std"]

    # Percentile ranking
    df["monsoon_percentile"] = df.groupby("SUBDIVISION")[
        "monsoon_rainfall"
    ].rank(pct=True)

    return df
