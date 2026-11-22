"""Step 4 + 9: compute the four classical centralities per snapshot, min-max
normalize them within each snapshot, and compute month-over-month deltas.

Degree centrality is unweighted (matches the original descriptive study).
Closeness/betweenness use `distance = 1 - jaccard` as edge length (higher
Jaccard similarity => shorter distance => stronger connection). Eigenvector
centrality uses the raw Jaccard weight as connection strength.

Parallelized with multiprocessing.Pool since each snapshot is independent.
Each worker is restricted to a single BLAS thread -- otherwise every worker's
numpy calls (e.g. the dense eigenvector fallback) try to multithread on top
of an already-parallel process pool and the machine oversubscribes its cores.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pickle
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

from . import config, graph_construction


def _minmax(d: dict) -> dict:
    if not d:
        return d
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _compute_one(args):
    month, G = args
    n = G.number_of_nodes()
    if n == 0:
        return month, pd.DataFrame(columns=["video_id"] + config.CENTRALITY_COLS)

    degree = nx.degree_centrality(G)
    closeness = nx.closeness_centrality(G, distance="distance")
    try:
        eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000, tol=1e-5)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")

    if n > 300:
        # exact betweenness is O(V*E); approximate with a pivot sample for
        # the dense keyword-overlap graphs typical of a monthly snapshot
        k = min(200, n)
        betweenness = nx.betweenness_centrality(G, k=k, weight="distance", seed=config.SEED)
    else:
        betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

    degree = _minmax(degree)
    closeness = _minmax(closeness)
    betweenness = _minmax(betweenness)
    eigenvector = _minmax(eigenvector)

    rows = [
        {
            "video_id": v,
            "degree": degree[v],
            "closeness": closeness[v],
            "betweenness": betweenness[v],
            "eigenvector": eigenvector[v],
        }
        for v in G.nodes()
    ]
    df = pd.DataFrame(rows)
    df["month"] = month
    return month, df


def compute_region_centrality(region: str, n_workers: int = 16) -> pd.DataFrame:
    graphs = graph_construction.build_and_cache(region)
    items = sorted(graphs.items())
    with Pool(n_workers) as pool:
        results = pool.map(_compute_one, items)
    frames = [df for _, df in sorted(results, key=lambda x: x[0])]
    out = pd.concat(frames, ignore_index=True)
    out["region"] = region

    # day-over-day delta per video (only valid when the previous row is
    # literally the immediately preceding calendar day, not just the
    # previous row for that video which may follow a multi-day gap).
    # NOTE: pd.PeriodIndex subtraction returns DateOffset objects rather
    # than integers in this pandas version, so `(p1 - p2) == 1` is silently
    # always False -- use plain datetime day-differences instead.
    out = out.sort_values(["video_id", "month"]).reset_index(drop=True)
    out["_date"] = pd.to_datetime(out["month"])
    grouped = out.groupby("video_id")
    prev_date = grouped["_date"].shift(1)
    is_consecutive = (out["_date"] - prev_date).dt.days == 1
    for c in config.CENTRALITY_COLS:
        prev = grouped[c].shift(1)
        delta = out[c] - prev
        out[f"delta_{c}"] = np.where(is_consecutive, delta, 0.0)
    out = out.drop(columns=["_date"])

    out["sis"] = out[config.CENTRALITY_COLS].mean(axis=1)
    return out.sort_values(["month", "video_id"]).reset_index(drop=True)


def build_and_cache(region: str, force: bool = False) -> pd.DataFrame:
    out_path = config.CACHE / f"centrality_{region}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    df = compute_region_centrality(region)
    df.to_parquet(out_path, index=False)
    return df


if __name__ == "__main__":
    import time

    for r in config.REGIONS:
        t0 = time.time()
        df = build_and_cache(r, force=True)
        print(r, df.shape, f"{time.time() - t0:.1f}s")
