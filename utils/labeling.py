import pandas as pd

def add_flood_label(df: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """
    Adds a binary flood label:
    1 = High flood risk year
    0 = Normal year

    threshold: percentile cutoff for extreme rainfall
    """
    df = df.copy()

    df["flood_risk"] = (
        df.groupby("SUBDIVISION")["monsoon_rainfall"]
        .transform(lambda x: x > x.quantile(threshold))
        .astype(int)
    )

    return df
