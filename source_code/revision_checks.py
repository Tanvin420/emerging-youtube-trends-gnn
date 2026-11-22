"""CPU-only cross-checks for the ICCIT revision (no torch needed).

Rebuilds labels/masks directly from the parquet caches, mirroring
build_dataset.py and persistence_baseline.py, and
  1. reproduces the published persistence baselines (status F1 0.836 /
     AUC 0.961, n = 81,760; emergence AUC 0.876, n = 65,310) as a check
     that the reconstruction matches the pipeline exactly;
  2. computes the persistence baseline for the graph-independent
     view-count target (view_labels.py): today's top-20% view status
     carried forward, scored by today's view count;
  3. counts labeled node-days per split (t+1), daily graph sizes, and
     same-channel co-trending pairs per day.

Run:  python -m source_code.revision_checks
Output: results/revision_checks_h1.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data_cache"
RESULTS = ROOT / "results"
REGIONS = ["US", "GB", "CA", "IN"]
TRAIN_END, VAL_END = "2021-11-30", "2022-01-15"   # as in config.py


def _metrics(y, score, pred):
    y, score, pred = map(np.asarray, (y, score, pred))
    k = int(y.sum())
    top = np.argsort(-score)[:k]
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, score), pr_auc=average_precision_score(y, score),
                precision_at_k=float(y[top].mean()), n=int(len(y)), pos_rate=float(y.mean()))


def _load(region):
    c = pd.read_parquet(CACHE / f"centrality_{region}.parquet")
    v = pd.read_parquet(CACHE / f"processed_videos_{region}.parquet")[
        ["video_id", "month", "max_view_count", "channelId"]]
    df = c.merge(v, on=["video_id", "month"], how="left")
    g = df.groupby("month")
    df["top20"] = (df["sis"] >= g["sis"].transform(lambda s: s.quantile(0.8))).astype(int)
    df["view_top20"] = (df["max_view_count"] >=
                        g["max_view_count"].transform(lambda s: s.quantile(0.8))).astype(int)
    return df


def main():
    rows = {k: [] for k in ["status", "emergence", "emergence_incl_departures", "view_target"]}
    counts, sizes, channel_pairs = {}, [], []
    for region in REGIONS:
        df = _load(region)
        days = sorted(df["month"].unique())
        by_day = {d: grp.set_index("video_id") for d, grp in df.groupby("month")}
        for d in days:
            day = by_day[d]
            sizes.append(dict(region=region, nodes=len(day)))
            per_ch = day.groupby("channelId").size()
            channel_pairs.append(int((per_ch * (per_ch - 1) // 2).sum()))
        for i, d in enumerate(days[:-1]):            # h = 1: next available snapshot
            cur, fut = by_day[d], by_day[days[i + 1]]
            present = cur.index.isin(fut.index)
            split = "train" if d <= TRAIN_END else ("val" if d <= VAL_END else "test")
            y = fut["top20"].reindex(cur.index).fillna(0).to_numpy()
            vy = fut["view_top20"].reindex(cur.index).fillna(0).to_numpy()
            low = cur["top20"].to_numpy() == 0
            emerge = present & low
            c = counts.setdefault(split, dict(status_n=0, status_pos=0, emergence_n=0,
                                              emergence_pos=0, low_n=0, low_departed=0))
            c["status_n"] += int(present.sum()); c["status_pos"] += int(y[present].sum())
            c["emergence_n"] += int(emerge.sum()); c["emergence_pos"] += int(y[emerge].sum())
            c["low_n"] += int(low.sum()); c["low_departed"] += int((low & ~present).sum())
            if split != "test":
                continue
            sis = cur["sis"].to_numpy()
            pred = (sis >= np.quantile(sis, 0.8)).astype(int)   # == current top-20%
            for name, m in [("status", present), ("emergence", emerge),
                            ("emergence_incl_departures", low)]:
                rows[name].append((y[m], sis[m], pred[m]))
            rows["view_target"].append((vy[present], cur["max_view_count"].to_numpy(float)[present],
                                        cur["view_top20"].to_numpy()[present]))

    out = {"persistence_baselines_h1": {
        k: _metrics(*(np.concatenate(z) for z in zip(*v))) for k, v in rows.items()}}
    for c in counts.values():
        c["emergence_pos_rate"] = c["emergence_pos"] / c["emergence_n"]
        c["low_departed_frac"] = c["low_departed"] / c["low_n"]
    out["split_counts_h1"] = counts
    s = pd.DataFrame(sizes)
    out["daily_graph_nodes"] = dict(min=int(s.nodes.min()), max=int(s.nodes.max()),
                                    mean_by_region=s.groupby("region").nodes.mean().round(1).to_dict(),
                                    snapshots_over_300_nodes=int((s.nodes > 300).sum()))
    out["same_channel_pairs_per_day_mean"] = float(np.mean(channel_pairs))
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "revision_checks_h1.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    for k, m in out["persistence_baselines_h1"].items():
        print(f"{k:27s} n={m['n']:6d} F1={m['f1']:.3f} AUC={m['auc']:.3f} PR-AUC={m['pr_auc']:.3f}")
    print("saved", path)


if __name__ == "__main__":
    main()
