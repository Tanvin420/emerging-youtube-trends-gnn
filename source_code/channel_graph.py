"""Same-channel co-trending edges: a second, structurally distinct edge
type alongside the keyword-overlap graph (graph_construction.py).

Two videos get an edge iff they share a channelId and are both trending in
the same daily snapshot. This is deliberately *not* derived from text or
centrality at all -- creator identity is invisible to Jaccard-over-TF-IDF
keywords, so unlike CGA-GAT's attention-bias term (found to be redundant
with centrality already available as a plain feature, see
run_experiments.run_centrality_ablation), a channel edge carries
information the keyword graph and its derived centrality structurally
cannot see. Most channels contribute only one trending video per day, so
this graph is naturally sparse; no Jaccard/threshold tuning is needed
since co-occurrence is binary.
"""
MAX_CHANNEL_CLIQUE = 30  # safety cap: skip a channel with an unusually
# large number of same-day trending videos rather than adding an O(n^2)
# clique for it (mirrors graph_construction.MAX_POSTING_LIST's role).


def build_snapshot_channel_edges(node_ids, channel_ids):
    """node_ids: ordered list of video ids for one snapshot, in the same
    order as that snapshot's node feature matrix. channel_ids: aligned
    sequence of each video's channelId (same length/order as node_ids).
    Returns (edge_src, edge_dst) as parallel 0-based position lists (both
    directions included, matching build_dataset.py's keyword-edge
    convention) suitable for a torch.tensor([edge_src, edge_dst])
    edge_index.
    """
    by_channel = {}
    for pos, ch in enumerate(channel_ids):
        if ch is None or ch == "" or ch != ch:  # ch != ch catches NaN
            continue
        by_channel.setdefault(ch, []).append(pos)

    edge_src, edge_dst = [], []
    for positions in by_channel.values():
        n = len(positions)
        if n < 2 or n > MAX_CHANNEL_CLIQUE:
            continue
        for i in range(n):
            a = positions[i]
            for j in range(i + 1, n):
                b = positions[j]
                edge_src += [a, b]
                edge_dst += [b, a]
    return edge_src, edge_dst
