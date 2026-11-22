"""Independent popularity target: whether a video's view count is in the
top 20% of that day's regional trending list. Computed exactly like
sis_labels.add_top20_label (same percentile-threshold mechanic) but on
raw view count instead of the centrality-derived SIS, so it shares no
construction with the four centralities used as model inputs. Exists to
test a real circularity concern: the paper's main label (top-20% SIS) is
the mean of the same four centralities given to the model as features, so
"predicts future top-20% status" could just mean "predicts future
centrality from current centrality." A model that predicts future
view-based popularity from current graph/centrality features is answering
a genuinely different, non-circular question.
"""
import pandas as pd

from . import config


def add_view_top20_label(df: pd.DataFrame, view_col: str = "max_view_count") -> pd.DataFrame:
    df = df.copy()

    def _flag(group):
        thresh = group[view_col].quantile(config.TOP_PERCENTILE)
        return (group[view_col] >= thresh).astype(int)

    df["view_top20"] = df.groupby(["region", "month"], group_keys=False).apply(_flag)
    return df
