"""Step 1-2: load raw regional CSVs, clean, build monthly video-level records.

Caches one parquet file per region: processed_videos_{region}.parquet
Columns: region, video_id, month, title, description, tags, categoryId,
         channelId, view_count, publishedAt, last_trending_date, clean_text
"""
import pandas as pd

from . import config
from .text_utils import clean_text

RAW_COLS = [
    "video_id", "title", "publishedAt", "channelId", "categoryId",
    "trending_date", "tags", "view_count", "description",
]


def load_region_raw(region: str) -> pd.DataFrame:
    path = config.DATA_RAW / f"{region}_youtube_trending_data.csv"
    df = pd.read_csv(path, usecols=RAW_COLS, low_memory=False)
    df["trending_date"] = pd.to_datetime(df["trending_date"], errors="coerce", utc=True)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    df = df.dropna(subset=["video_id", "trending_date"])
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["tags"] = df["tags"].fillna("").replace("[none]", "")
    df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0)
    # Snapshot granularity: a video's median trending run in this dataset is
    # only ~5 days (99th pct ~10 days), so monthly bins leave almost no video
    # persisting across snapshots beyond a 1-step gap, making t+2/t+3
    # prediction ill-posed. Daily bins match the data's real churn rate and
    # keep meaningful node overlap out to t+3. Field is still called "month"
    # to avoid renaming it through the rest of the pipeline.
    df["month"] = df["trending_date"].dt.strftime("%Y-%m-%d")
    return df


def build_video_month_table(region: str) -> pd.DataFrame:
    df = load_region_raw(region)

    # aggregate to one row per (video_id, day): take the latest trending
    # record that day for text/category, and the max view_count observed
    # that day (view_count only grows over time on a video).
    df = df.sort_values("trending_date")
    last = df.groupby(["video_id", "month"], as_index=False).last()
    max_views = df.groupby(["video_id", "month"])["view_count"].max().rename("max_view_count")
    out = last.merge(max_views, on=["video_id", "month"])
    out = out.rename(columns={"trending_date": "last_trending_date", "view_count": "_drop"})
    out = out.drop(columns=["_drop"])
    out["region"] = region

    text = (out["title"] + " " + out["description"] + " " + out["tags"].str.replace("|", " ", regex=False))
    out["clean_text"] = text.map(clean_text)

    cols = [
        "region", "video_id", "month", "title", "description", "tags",
        "categoryId", "channelId", "max_view_count", "publishedAt",
        "last_trending_date", "clean_text",
    ]
    return out[cols].sort_values(["month", "video_id"]).reset_index(drop=True)


def build_and_cache(region: str, force: bool = False) -> pd.DataFrame:
    out_path = config.CACHE / f"processed_videos_{region}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    table = build_video_month_table(region)
    table.to_parquet(out_path, index=False)
    return table


if __name__ == "__main__":
    for r in config.REGIONS:
        t = build_and_cache(r, force=True)
        print(r, t.shape, t["month"].nunique(), "months")
