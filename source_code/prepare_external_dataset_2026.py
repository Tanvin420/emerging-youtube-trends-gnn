"""Cross-dataset validation, step 0 (era 2: 2025-2026): adapt the live,
actively-updated "YouTube Trending Dataset (Updated Daily)" (110 countries,
GitHub Actions + YouTube Data API, CC0) to the raw schema
`data_prep.load_region_raw` expects, exactly like prepare_external_dataset.py
does for the 2017-2018 dataset -- so the rest of the pipeline runs on it
unmodified. Source: https://github.com/sheb1lmsp/youtube-trending-analytics
(mirrored to Kaggle as shebilmsp/youtube-trending-dataset-updated-daily).

Unlike the 2020-2022 training window this project uses, this dataset starts
2025-12-10 and is updated through the current day -- i.e. it postdates the
entire training window, the mirror image of the 2018 dataset which predates
it. One CSV per country per day (data/videos/country=XX/year=YYYY/month=MM/
trending_XX_YYYY-MM-DD.csv); no per-video description field (dropped from
clean_text, title+tags only for this era).

Usage:
    # one-time: shallow, blobless, sparse clone (repo is ~900MB across 110
    # countries; this pulls only the 6 countries below, ~260MB)
    git clone --filter=blob:none --no-checkout --depth 1 \\
        https://github.com/sheb1lmsp/youtube-trending-analytics.git \\
        dataset/external_2026/repo
    cd dataset/external_2026/repo
    git sparse-checkout init --cone
    git sparse-checkout set data/videos/country=US data/videos/country=GB \\
        data/videos/country=CA data/videos/country=IN data/videos/country=DE \\
        data/videos/country=FR
    git checkout main

    python -m source_code.prepare_external_dataset_2026
"""
import sys

import pandas as pd

from . import config

RAW_DIR = config.ROOT / "dataset" / "external_2026" / "repo" / "data" / "videos"

# maps our external region code -> source country code
EXTERNAL_TO_SOURCE_CODE = {"CA26": "CA", "DE26": "DE", "FR26": "FR",
                            "GB26": "GB", "IN26": "IN", "US26": "US"}


def convert_region(region: str, force: bool = False) -> pd.DataFrame:
    code = EXTERNAL_TO_SOURCE_CODE[region]
    out_path = config.DATA_RAW / f"{region}_youtube_trending_data.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    files = sorted((RAW_DIR / f"country={code}").glob("year=*/month=*/trending_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no source files under {RAW_DIR / f'country={code}'} -- run the "
            "git sparse-checkout in this module's docstring first"
        )

    frames = []
    for f in files:
        d = pd.read_csv(f)
        # trending day is the filename's date, not `fetched_at` (a same-day
        # UTC timestamp of when the daily job ran) -- consistent with how
        # every other region's trending_date is a plain calendar day.
        d["trending_date"] = f.stem.rsplit("_", 1)[1]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame({
        "video_id": df["video_id"],
        "title": df["title"],
        "publishedAt": df["published_at"],
        "channelId": df["channel_id"],
        "categoryId": df["category_id"],
        "trending_date": df["trending_date"],
        # this schema separates tags with ", " rather than the "|" every
        # other region in this project uses -- normalize so data_prep.py's
        # tags.str.replace("|", " ") behaves the same way here.
        "tags": df["tags"].fillna("").str.replace(", ", "|", regex=False).str.replace(",", "|", regex=False),
        "view_count": df["views"],
        "description": "",  # not present in this schema
    })
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"{region}: wrote {len(out)} rows -> {out_path}")
    return out


def main():
    for region in EXTERNAL_TO_SOURCE_CODE:
        convert_region(region, force=("--force" in sys.argv))


if __name__ == "__main__":
    sys.exit(main())
