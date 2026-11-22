"""Shared paths and constants for the Centrality-Guided GAT pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repository root
DATA_RAW = ROOT / "dataset"                  # downloaded trending CSVs
CACHE = ROOT / "data_cache"                  # intermediate files written by the pipeline
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for d in (CACHE, RESULTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

REGIONS = ["US", "GB", "CA", "IN"]
REGION_NAMES = {"US": "USA", "GB": "UK", "CA": "Canada", "IN": "India"}

# Cross-dataset validation: two independent datasets/eras never used for
# training or model selection -- evaluation only, via
# source_code/prepare_external_dataset.py (2018) / source_code/prepare_external_dataset_2026.py
# (2026) and source_code/cross_dataset.py. "2018" predates REGIONS' 2020-2022 window
# (a different collection instance, the classic Kaggle "Trending YouTube
# Video Statistics" dataset); "2026" postdates it entirely (a live,
# actively-updated 110-country dataset), so together they bracket the
# training window on both sides -- see EXTERNAL_ERAS below.
EXTERNAL_REGIONS = ["CA18", "DE18", "FR18", "GB18", "US18"]
EXTERNAL_REGION_NAMES = {
    "CA18": "Canada '18", "DE18": "Germany '18", "FR18": "France '18",
    "GB18": "UK '18", "US18": "USA '18",
}

EXTERNAL_REGIONS_2026 = ["CA26", "DE26", "FR26", "GB26", "IN26", "US26"]
EXTERNAL_REGION_NAMES_2026 = {
    "CA26": "Canada '26", "DE26": "Germany '26", "FR26": "France '26",
    "GB26": "UK '26", "IN26": "India '26", "US26": "USA '26",
}

# era label -> (regions, region_names, human-readable date range) for the
# cross-era comparison in source_code/cross_dataset.py
EXTERNAL_ERAS = {
    "2018": (EXTERNAL_REGIONS, EXTERNAL_REGION_NAMES, "2017-11-14 to 2018-01-05"),
    "2026": (EXTERNAL_REGIONS_2026, EXTERNAL_REGION_NAMES_2026, "2025-12-10 to 2026-09-14"),
}

# Keyword extraction
TOP_K_KEYWORDS = 15
MIN_DF_KEYWORDS = 2
MAX_KEYWORD_VOCAB = 20000

# Graph construction
MIN_EDGE_WEIGHT = 1e-6  # Jaccard threshold below which no edge is kept
MIN_SHARED_KEYWORDS = 1

# Centrality
CENTRALITY_COLS = ["degree", "closeness", "betweenness", "eigenvector"]
DELTA_COLS = [f"delta_{c}" for c in CENTRALITY_COLS]

# SIS / label
TOP_PERCENTILE = 0.80

# Node feature layout, as concatenated in build_dataset.py:
# [0:4]=current centrality, [4:8]=delta centrality, [8]=log views,
# [9:9+len(CATEGORY_IDS)]=category one-hot, rest=text embedding.
# CENTRALITY_FEATURE_SLICE marks the columns that duplicate what CGA-GAT's
# attention bias also receives via the `cent` tensor -- used by the
# feature-ablation study in train_eval.py to isolate whether the attention
# bias adds anything beyond having centrality available as a plain feature.
CENTRALITY_FEATURE_SLICE = slice(0, len(CENTRALITY_COLS))
DELTA_FEATURE_SLICE = slice(len(CENTRALITY_COLS), 2 * len(CENTRALITY_COLS))

# Fixed category vocabulary observed across US/GB/CA/IN trending data
CATEGORY_IDS = [1, 2, 10, 15, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29]

# Remaining feature-block slices, for the fuller feature-ablation hierarchy
# in train_eval.py (text / views / category / centrality contribution study).
_VIEWS_START = DELTA_FEATURE_SLICE.stop
VIEWS_FEATURE_SLICE = slice(_VIEWS_START, _VIEWS_START + 1)
CATEGORY_FEATURE_SLICE = slice(VIEWS_FEATURE_SLICE.stop, VIEWS_FEATURE_SLICE.stop + len(CATEGORY_IDS))
TEXT_FEATURE_SLICE = slice(CATEGORY_FEATURE_SLICE.stop, CATEGORY_FEATURE_SLICE.stop + 384)  # EMBED_DIM

# Snapshot horizons (in days) to evaluate -- see data_prep.py for why the
# snapshot unit is a day rather than a month.
HORIZONS = [1, 2, 3]

# Temporal split boundaries (ISO date strings, inclusive), applied per region.
# Data runs 2020-08-12..2022-05-29 (~656 days). Cutoffs chosen so the test
# window still has days with t+3 <= the last available day, otherwise the
# t+3 horizon would have zero usable test snapshots.
TRAIN_END = "2021-11-30"   # train: snapshots <= this day (~475 days)
VAL_END = "2022-01-15"     # val: days in (TRAIN_END, VAL_END] (~46 days)
# test: everything after VAL_END up to the last usable snapshot (~135 days)

# Embeddings
SENTENCE_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Model / training
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_GAT_LAYERS = 2
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 5e-4
EPOCHS = 150
PATIENCE = 20
SEED = 42
