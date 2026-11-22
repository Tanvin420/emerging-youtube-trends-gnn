"""Step 3b: build a weighted keyword-overlap graph for each (region, month)
snapshot, using Jaccard similarity of keyword sets as edge weight.

Uses an inverted (keyword -> video) index to generate candidate pairs
instead of an O(n^2) scan, matching the "prefer sparse representations"
requirement in the plan.
"""
import pickle
from collections import Counter, defaultdict

import networkx as nx

from . import config, keyword_extraction

MAX_POSTING_LIST = 400  # cap on how many videos may share one keyword


def build_month_graph(month_keywords: dict) -> nx.Graph:
    vids = list(month_keywords.keys())
    sizes = {v: len(month_keywords[v]) for v in vids}

    postings = defaultdict(list)
    for v, kws in month_keywords.items():
        for k in kws:
            postings[k].append(v)

    pair_counts = Counter()
    for plist in postings.values():
        n = len(plist)
        if n < 2 or n > MAX_POSTING_LIST:
            continue
        for i in range(n):
            a = plist[i]
            for j in range(i + 1, n):
                b = plist[j]
                key = (a, b) if a < b else (b, a)
                pair_counts[key] += 1

    G = nx.Graph()
    G.add_nodes_from(vids)
    for (a, b), inter in pair_counts.items():
        if inter < config.MIN_SHARED_KEYWORDS:
            continue
        union = sizes[a] + sizes[b] - inter
        if union <= 0:
            continue
        jac = inter / union
        if jac >= config.MIN_EDGE_WEIGHT:
            G.add_edge(a, b, weight=jac, distance=1.0 - jac + 1e-6)
    return G


def build_region_graphs(region: str) -> dict:
    kw = keyword_extraction.build_and_cache(region)
    return {month: build_month_graph(m_kw) for month, m_kw in kw.items()}


def build_and_cache(region: str, force: bool = False) -> dict:
    out_path = config.CACHE / f"graphs_{region}.pkl"
    if out_path.exists() and not force:
        with open(out_path, "rb") as f:
            return pickle.load(f)
    graphs = build_region_graphs(region)
    with open(out_path, "wb") as f:
        pickle.dump(graphs, f)
    return graphs


if __name__ == "__main__":
    import time

    for r in config.REGIONS:
        t0 = time.time()
        graphs = build_and_cache(r, force=True)
        dt = time.time() - t0
        n_nodes = sum(g.number_of_nodes() for g in graphs.values())
        n_edges = sum(g.number_of_edges() for g in graphs.values())
        print(f"{r}: {len(graphs)} snapshots, {n_nodes} node-months, "
              f"{n_edges} edges total, {dt:.1f}s")
