# Dataset folder (`dataset/`)

The pipeline reads one raw CSV per region from this folder (`config.DATA_RAW`). The files are not stored in the
repository because they are large; download them from the sources below.

## Dataset sources

| Role in the paper | Dataset | Period used | Regions | Full link |
|---|---|---|---|---|
| Training, validation, and test | YouTube Trending Video Dataset (updated daily), by rsrishav | 2020-08-12 to 2022-05-29 | US, GB, CA, IN | https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset |
| Zero-shot test, Era 2018 | Trending YouTube Video Statistics, by Mitchell J (datasnaek) | 2017-11-14 to 2018-01-05 | CA, DE, FR, GB, US | https://www.kaggle.com/datasets/datasnaek/youtube-new |
| Era 2018, mirror used by our download script | Public GitHub mirror of the Kaggle files | same as above | same as above | https://github.com/CloudChaoszero/Youtube-TrendsNMore/tree/master/Data |
| Zero-shot test, Era 2026 | YouTube Trending Dataset (updated daily), by sheb1lmsp | 2025-12-10 to 2026-09-14 | CA, DE, FR, GB, IN, US | https://github.com/sheb1lmsp/youtube-trending-analytics |
| Era 2026, Kaggle mirror | same dataset on Kaggle | same as above | same as above | https://www.kaggle.com/datasets/shebilmsp/youtube-trending-dataset-updated-daily |

## Expected files

All files follow the pattern `<REGION>_youtube_trending_data.csv`.

| Files | How to obtain them |
|---|---|
| `US_youtube_trending_data.csv`, `GB_...`, `CA_...`, `IN_...` | Download from the rsrishav Kaggle page and copy them here. Keep only rows with `trending_date` from 2020-08-12 to 2022-05-29 (the dataset kept growing after our download; see the trimming snippet in the main README). |
| `CA18_...`, `DE18_...`, `FR18_...`, `GB18_...`, `US18_...` | Run `python -m source_code.prepare_external_dataset`. It downloads the 2017-18 files from the GitHub mirror (for example https://raw.githubusercontent.com/CloudChaoszero/Youtube-TrendsNMore/master/Data/USvideos.csv) into `dataset/external_2018/` and writes the converted files here. |
| `CA26_...`, `DE26_...`, `FR26_...`, `GB26_...`, `IN26_...`, `US26_...` | Sparse-checkout the six countries from https://github.com/sheb1lmsp/youtube-trending-analytics into `dataset/external_2026/repo` (commands in the main README), then run `python -m source_code.prepare_external_dataset_2026`. Keep only 2025-12-10 to 2026-09-14 to match the paper. |

Please respect the terms of use of each original dataset and cite them if you use them.
