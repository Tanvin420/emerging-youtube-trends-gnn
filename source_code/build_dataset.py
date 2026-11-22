"""Step 6: assemble a PyTorch Geometric Data object per (region, month)
snapshot: node features at t, keyword-overlap graph at t, and binary future
top-20%-structural-influence labels at t+1, t+2, t+3.

All input features (centrality, deltas, view count, category, text) come
from month t or earlier; labels come strictly from month t+h. No future
information leaks into the inputs (see plan Sec. 18).
"""
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from . import channel_graph, config, centrality, data_prep, embeddings, graph_construction, sis_labels, view_labels

NUM_CENTRALITY = len(config.CENTRALITY_COLS)


def _load_combined(region: str) -> pd.DataFrame:
    cent = sis_labels.add_top20_label(centrality.build_and_cache(region))
    videos = data_prep.build_and_cache(region)[["region", "month", "video_id", "max_view_count", "categoryId", "channelId"]]
    videos = view_labels.add_view_top20_label(videos)
    emb_df = embeddings.get_region_embeddings(region)[["month", "video_id", "emb"]]

    combined = (
        cent.merge(videos.drop(columns=["region"]), on=["month", "video_id"], how="left")
            .merge(emb_df, on=["month", "video_id"], how="left")
    )
    return combined


def build_region_snapshots(region: str) -> dict:
    graphs = graph_construction.build_and_cache(region)
    combined = _load_combined(region)
    month_groups = {m: g.set_index("video_id") for m, g in combined.groupby("month")}

    months = sorted(graphs.keys())
    month_to_idx = {m: i for i, m in enumerate(months)}

    snapshots = {}
    for month in months:
        G = graphs[month]
        node_ids = sorted(G.nodes())
        if not node_ids:
            continue
        node_pos = {v: i for i, v in enumerate(node_ids)}

        mg = month_groups[month].reindex(node_ids)
        c = mg[config.CENTRALITY_COLS].to_numpy(dtype=np.float32)
        d = mg[config.DELTA_COLS].to_numpy(dtype=np.float32)
        sis_vals = mg["sis"].to_numpy(dtype=np.float32)
        log_views = np.log1p(mg["max_view_count"].to_numpy(dtype=np.float32))

        cat_ids = mg["categoryId"].to_numpy()
        cat_oh = np.zeros((len(node_ids), len(config.CATEGORY_IDS)), dtype=np.float32)
        for ci, cat in enumerate(config.CATEGORY_IDS):
            cat_oh[:, ci] = (cat_ids == cat)

        emb_stack = np.stack(mg["emb"].to_numpy()).astype(np.float32)

        feats = np.concatenate(
            [c, d, log_views[:, None], cat_oh, emb_stack], axis=1
        ).astype(np.float32)

        edge_src, edge_dst, edge_w = [], [], []
        for u, v, edata in G.edges(data=True):
            iu, iv = node_pos[u], node_pos[v]
            w = edata["weight"]
            edge_src += [iu, iv]
            edge_dst += [iv, iu]
            edge_w += [w, w]

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_weight = torch.tensor(edge_w, dtype=torch.float32)

        # Second, structurally distinct edge type: same-channel co-trending
        # videos (channel_graph.py). Kept separate from edge_index/edge_attr
        # above rather than merged into one graph, so models can choose to
        # aggregate over each relation with its own attention head
        # (DualSPCGAGATConv in models.py) instead of conflating "shares
        # keywords" and "shares a creator" into a single edge weight.
        ch_src, ch_dst = channel_graph.build_snapshot_channel_edges(
            node_ids, mg["channelId"].tolist())
        edge_index_channel = torch.tensor([ch_src, ch_dst], dtype=torch.long)

        current_top_np = mg["top20"].fillna(0.0).to_numpy(dtype=np.float32)

        data_obj = Data(x=torch.tensor(feats), edge_index=edge_index, edge_attr=edge_weight)
        data_obj.edge_index_channel = edge_index_channel
        data_obj.cent = torch.tensor(c, dtype=torch.float32)
        data_obj.sis = torch.tensor(sis_vals, dtype=torch.float32)
        data_obj.current_top = torch.tensor(current_top_np)
        data_obj.video_id = node_ids
        data_obj.region = region
        data_obj.month = month

        for h in config.HORIZONS:
            future_idx = month_to_idx[month] + h
            if future_idx < len(months):
                future_month = months[future_idx]
                fmg = month_groups[future_month].reindex(node_ids)
                present_np = fmg["top20"].notna().to_numpy()
                mask_np = present_np
                y_np = fmg["top20"].fillna(0.0).to_numpy(dtype=np.float32)
                # independent, non-centrality popularity target (view_labels.py):
                # same presence-based mask, but the label is future view-count
                # top-20% instead of future SIS top-20% -- tests whether current
                # graph/centrality features predict future *popularity*, not
                # just future *centrality* (the label they were built from).
                view_y_np = fmg["view_top20"].fillna(0.0).to_numpy(dtype=np.float32)
            else:
                present_np = np.zeros(len(node_ids), dtype=bool)
                mask_np = present_np
                y_np = np.zeros(len(node_ids), dtype=np.float32)
                view_y_np = np.zeros(len(node_ids), dtype=np.float32)
            setattr(data_obj, f"y{h}", torch.tensor(y_np))
            setattr(data_obj, f"mask{h}", torch.tensor(mask_np))
            setattr(data_obj, f"view_y{h}", torch.tensor(view_y_np))
            # emergence task (original): eligible population is nodes NOT
            # already top-20% at t -- a video that's already influential
            # can't "emerge" again. This variant additionally requires the
            # video to still be PRESENT at t+h, which silently drops videos
            # that stop trending -- a survivorship bias (a disappeared video
            # is simply excluded rather than counted as "did not emerge").
            emerge_mask_np = mask_np & (current_top_np < 0.5)
            setattr(data_obj, f"emerge_mask{h}", torch.tensor(emerge_mask_np))
            # survivorship-corrected variant: eligibility no longer requires
            # future presence, so a disappeared video is scored as a
            # (correct-by-construction) negative -- "did not emerge" --
            # instead of being dropped from evaluation entirely.
            emerge_mask_all_np = current_top_np < 0.5
            setattr(data_obj, f"emerge_mask_all{h}", torch.tensor(emerge_mask_all_np))
            # descriptive 3-way outcome for the same eligible (low-at-t)
            # population: 0 = disappeared by t+h, 1 = present but still low,
            # 2 = present and now top-20% -- lets us report what fraction of
            # "low" videos take each path, not just the binary emerge/not.
            outcome_np = np.where(~present_np, 0, np.where(y_np > 0.5, 2, 1)).astype(np.float32)
            setattr(data_obj, f"emerge_outcome3_{h}", torch.tensor(outcome_np))

        snapshots[month] = data_obj
    return snapshots


def build_and_cache(region: str, force: bool = False):
    out_path = config.CACHE / f"snapshots_{region}.pkl"
    if out_path.exists() and not force:
        with open(out_path, "rb") as f:
            return pickle.load(f)
    snaps = build_region_snapshots(region)
    with open(out_path, "wb") as f:
        pickle.dump(snaps, f)
    return snaps


if __name__ == "__main__":
    for r in config.REGIONS:
        snaps = build_and_cache(r, force=True)
        m0 = sorted(snaps.keys())[0]
        d = snaps[m0]
        print(r, len(snaps), "snapshots; example", m0, d)
