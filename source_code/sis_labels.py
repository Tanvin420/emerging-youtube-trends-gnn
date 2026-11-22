"""Step 11: Structural Influence Score (SIS) and the top-20% binary label,
computed per (region, month) snapshot."""
import pandas as pd

from . import config


def add_top20_label(centrality_df: pd.DataFrame) -> pd.DataFrame:
    df = centrality_df.copy()

    def _flag(group):
        thresh = group["sis"].quantile(config.TOP_PERCENTILE)
        return (group["sis"] >= thresh).astype(int)

    df["top20"] = df.groupby(["region", "month"], group_keys=False).apply(_flag)
    return df
