"""Cross-dataset validation, step 0: adapt the classic 2017-2018 Kaggle
"Trending YouTube Video Statistics" dataset (Mitchell J / datasnaek -- the
direct predecessor of the rsrishav "YouTube Trending Video Dataset (updated
daily)" collection this project already uses) to the exact raw schema
`data_prep.load_region_raw` expects, so the rest of the pipeline (keyword
extraction, graph construction, centrality, embeddings, dataset assembly)
runs on it completely unmodified.

Source data (not redistributed here -- run this script to fetch+convert):
  https://raw.githubusercontent.com/CloudChaoszero/Youtube-TrendsNMore/master/Data/{CODE}videos.csv
This is a public mirror of the original Kaggle dataset (CA/DE/FR/GB/US,
2017-11-14 to 2018-01-05, ~200 trending videos/day/region). It predates our
main dataset by ~3 years and was collected via an earlier version of the
scraper/API, making it a genuine cross-dataset (not just cross-region) test:
different collection instance, different time period, two countries our
models never see in training (Germany, France) plus two they do (Canada,
UK) at an earlier point in the platform's history.

Usage:
    python -m source_code.prepare_external_dataset          # download + convert
    python -m source_code.prepare_external_dataset --no-download  # convert only
"""
import argparse
import sys

import pandas as pd

from . import config

RAW_DIR = config.ROOT / "dataset" / "external_2018"
SOURCE_URL = "https://raw.githubusercontent.com/CloudChaoszero/Youtube-TrendsNMore/master/Data/{code}videos.csv"

# maps our external region code -> classic-dataset country code
EXTERNAL_TO_SOURCE_CODE = {"CA18": "CA", "DE18": "DE", "FR18": "FR", "GB18": "GB", "US18": "US"}


def download_raw(force: bool = False) -> None:
    import urllib.request

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for region, code in EXTERNAL_TO_SOURCE_CODE.items():
        out_path = RAW_DIR / f"{code}videos_2018.csv"
        if out_path.exists() and not force:
            continue
        url = SOURCE_URL.format(code=code)
        print(f"downloading {url} -> {out_path}")
        urllib.request.urlretrieve(url, out_path)


def convert_region(region: str, force: bool = False) -> pd.DataFrame:
    """Reads dataset/external_2018/{code}videos_2018.csv and writes
    dataset/{region}_youtube_trending_data.csv in the same 9-column raw
    schema as the original regions (see data_prep.RAW_COLS)."""
    code = EXTERNAL_TO_SOURCE_CODE[region]
    out_path = config.DATA_RAW / f"{region}_youtube_trending_data.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    src_path = RAW_DIR / f"{code}videos_2018.csv"
    # the classic dataset's CSVs contain non-UTF-8 bytes in some titles/tags
    df = pd.read_csv(src_path, encoding="latin1")

    # trending_date is "YY.DD.MM" (e.g. "17.14.11" = 2017-11-14) here, vs.
    # full ISO datetimes in the dataset the pipeline was built for -- both
    # parse fine through data_prep's pd.to_datetime(errors="coerce", utc=True)
    # once normalized to an ISO date string.
    trending_date = pd.to_datetime(df["trending_date"], format="%y.%d.%m").dt.strftime("%Y-%m-%d")

    out = pd.DataFrame({
        "video_id": df["video_id"],
        "title": df["title"],
        "publishedAt": df["publish_time"],
        "channelId": df["channel_title"],  # no numeric channel id in this vintage; unused downstream
        "categoryId": df["category_id"],
        "trending_date": trending_date,
        "tags": df["tags"],
        "view_count": df["views"],
        "description": df["description"],
    })
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"{region}: wrote {len(out)} rows -> {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.no_download:
        download_raw(force=args.force)
    for region in EXTERNAL_TO_SOURCE_CODE:
        convert_region(region, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
