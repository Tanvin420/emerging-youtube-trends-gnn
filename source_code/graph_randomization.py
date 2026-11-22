"""Graph-structure sanity check: does the actual keyword-overlap topology
carry information beyond what's already summarized in the per-node
centrality features?

We keep every node's features (including its real centrality/delta/text/
view/category values) identical, and only replace each snapshot's edges
with a degree-preserving random rewiring (networkx double_edge_swap, the
standard configuration-model-style null model). If real-graph performance
clearly exceeds shuffled-graph performance, that's evidence the topology
itself -- not just the centrality summary statistics derived from it -- is
informative to message passing. If they're indistinguishable, message
passing isn't using anything beyond what's already in the node features.
"""
import copy
import multiprocessing as mp

import networkx as nx
import numpy as np
import torch

from . import build_dataset, config

# multiprocessing.Pool's default "fork" start method deadlocks here: by the
# time this runs, the calling process has already initialized a CUDA
# context (from GPU model training earlier in the same script), and CUDA
# is not fork-safe -- forked workers inherit a broken CUDA state and hang
# forever on the first allocation, even though this module itself never
# touches the GPU. "spawn" starts clean child interpreters instead.
_MP_CTX = mp.get_context("spawn")


def _randomize_data(data, seed):
    n = data.x.size(0)
    edge_index = data.edge_index
    # edges are stored in both directions; dedupe to an undirected edge list
    pairs = edge_index.t().tolist()
    edge_set = {tuple(sorted(e)) for e in pairs if e[0] != e[1]}

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_set)
    n_edges = G.number_of_edges()
    if n_edges >= 2:
        try:
            nx.double_edge_swap(G, nswap=n_edges * 2, max_tries=n_edges * 20, seed=seed)
        except nx.NetworkXAlgorithmError:
            pass  # graph too sparse/small to complete requested swaps; keep partial result

    new_edges = list(G.edges())
    src = [a for a, b in new_edges] + [b for a, b in new_edges]
    dst = [b for a, b in new_edges] + [a for a, b in new_edges]

    new_data = copy.copy(data)
    new_data.edge_index = torch.tensor([src, dst], dtype=torch.long)
    new_data.edge_attr = torch.ones(len(src), dtype=torch.float32)
    return new_data


def _randomize_one(args):
    day, data, seed = args
    return day, _randomize_data(data, seed=seed)


def randomize_all_snapshots(all_snaps, seed=42, n_workers=16):
    """Returns a new {region: {day: Data}} with every snapshot's edges
    degree-preserving-shuffled; node features/labels are untouched.
    Parallelized (networkx's double_edge_swap is pure Python and slow) with
    a single spawn-context pool shared across all regions -- "spawn" starts
    fresh interpreters (slow to boot, ~10s for 16 workers given torch/PyG
    imports), so reuse one pool rather than paying that cost per region."""
    jobs = []
    i = 0
    for region, snaps in all_snaps.items():
        for day, data in snaps.items():
            jobs.append((region, day, data, seed + i))
            i += 1

    with _MP_CTX.Pool(n_workers) as pool:
        results = pool.map(_randomize_one, [(day, data, s) for _, day, data, s in jobs])

    out = {region: {} for region in all_snaps}
    for (region, day, _, _), (out_day, out_data) in zip(jobs, results):
        out[region][out_day] = out_data
    return out


if __name__ == "__main__":
    from . import train_eval

    real_snaps = train_eval.load_all_snapshots(config.REGIONS)
    rand_snaps = randomize_all_snapshots(real_snaps, seed=42)

    d0 = next(iter(real_snaps[config.REGIONS[0]].values()))
    r0 = next(iter(rand_snaps[config.REGIONS[0]].values()))
    print("real edges:", d0.edge_index.size(1), "random edges:", r0.edge_index.size(1))
    print("features identical:", torch.equal(d0.x, r0.x))
