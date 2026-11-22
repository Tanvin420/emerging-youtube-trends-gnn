"""Step 17-19: temporal train/val/test split, training loop, and evaluation
metrics for the future-structural-influence prediction task.

Supports two task framings via `mask_attr`:
  - f"mask{h}"        : predict top-20% status at t+h over all nodes present
                         at both t and t+h (the original framing).
  - f"emerge_mask{h}" : predict top-20% status at t+h restricted to nodes
                         that are NOT already top-20% at t (the "genuine
                         emergence" framing -- see build_dataset.py).

Also supports a feature-ablation switch (`feature_ablation="no_centrality"`)
that zeroes the centrality columns of x, to separate "centrality available
as a plain feature" from "centrality injected into GAT's attention bias"
when comparing GAT and CGA-GAT -- otherwise every model already sees
centrality in x and the comparison is confounded.
"""
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              roc_auc_score)
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from . import build_dataset, config, models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NOTE: GPU ops (scatter/index_add_ in CGAGATConv, PyG's internal
    # aggregation) are non-deterministic by default -- confirmed empirically:
    # two runs with the same seed gave different F1/AUC (e.g. 0.1425 vs
    # 0.1439). torch.use_deterministic_algorithms(True) fixes this but was
    # measured to cause a >10x slowdown on this workload (20+ min with zero
    # completed runs vs ~90s/run normally), making it impractical for
    # multi-seed sweeps here. We accept GPU-level nondeterminism as part of
    # the "run-to-run variance" multi-seed runs are already measuring --
    # standard practice, and the headline gaps we report (e.g. MLP/GCN vs
    # GAT/CGA-GAT on the emergence task) are far larger than this noise.


def split_months(months):
    train = [m for m in months if m <= config.TRAIN_END]
    val = [m for m in months if config.TRAIN_END < m <= config.VAL_END]
    test = [m for m in months if m > config.VAL_END]
    return train, val, test


def load_all_snapshots(regions):
    """Returns {region: {month: Data}} using cached snapshot builds."""
    return {r: build_dataset.build_and_cache(r) for r in regions}


def collect_split_snapshots(all_snaps, regions, horizon, mask_attr=None):
    """Returns dict split -> list of (region, month, data) with a non-empty
    mask for this horizon/task (i.e., there's at least one eligible,
    labeled node in that snapshot)."""
    mask_attr = mask_attr or f"mask{horizon}"
    splits = {"train": [], "val": [], "test": []}
    for region in regions:
        snaps = all_snaps[region]
        months = sorted(snaps.keys())
        train_m, val_m, test_m = split_months(months)
        for split_name, month_list in [("train", train_m), ("val", val_m), ("test", test_m)]:
            for m in month_list:
                data = snaps[m]
                mask = getattr(data, mask_attr)
                if mask.sum().item() == 0:
                    continue
                splits[split_name].append((region, m, data))
    return splits


# Feature-block ablation hierarchy (Sec. "Where does the predictive power
# come from?"): each entry either zeroes out specific feature blocks
# ("zero") or keeps only specific blocks and zeroes everything else
# ("keep_only"). Slices refer to the column layout documented in config.py.
FEATURE_ABLATIONS = {
    "full": {"zero": []},
    "no_centrality": {"zero": [config.CENTRALITY_FEATURE_SLICE]},
    "no_delta": {"zero": [config.DELTA_FEATURE_SLICE]},
    "no_views": {"zero": [config.VIEWS_FEATURE_SLICE]},
    "no_category": {"zero": [config.CATEGORY_FEATURE_SLICE]},
    "no_text": {"zero": [config.TEXT_FEATURE_SLICE]},
    "centrality_only": {"keep_only": [config.CENTRALITY_FEATURE_SLICE, config.DELTA_FEATURE_SLICE]},
    "text_only": {"keep_only": [config.TEXT_FEATURE_SLICE]},
    "no_graph_derived": {"zero": [config.CENTRALITY_FEATURE_SLICE, config.DELTA_FEATURE_SLICE]},
}


def _apply_feature_ablation(x, feature_ablation):
    spec = FEATURE_ABLATIONS.get(feature_ablation)
    if spec is None:
        raise ValueError(f"unknown feature_ablation {feature_ablation}")
    if not spec.get("zero") and not spec.get("keep_only"):
        return x
    x = x.clone()
    if "keep_only" in spec:
        mask = torch.zeros(x.size(1), dtype=torch.bool, device=x.device)
        for sl in spec["keep_only"]:
            mask[sl] = True
        x[:, ~mask] = 0.0
    else:
        for sl in spec["zero"]:
            x[:, sl] = 0.0
    return x


def train_model(model_name, regions, horizon, mask_attr=None, y_attr=None, feature_ablation="full",
                 epochs=config.EPOCHS, lr=config.LR, weight_decay=config.WEIGHT_DECAY,
                 patience=config.PATIENCE, seed=config.SEED, verbose=False, all_snaps=None):
    mask_attr = mask_attr or f"mask{horizon}"
    y_attr = y_attr or f"y{horizon}"
    set_seed(seed)
    if all_snaps is None:
        all_snaps = load_all_snapshots(regions)
    splits = collect_split_snapshots(all_snaps, regions, horizon, mask_attr=mask_attr)

    example = splits["train"][0][2]
    in_dim = example.x.size(1)
    centrality_dim = example.cent.size(1)

    model = models.build_model(model_name, in_dim, centrality_dim=centrality_dim,
                                hidden=config.HIDDEN_DIM, heads=config.NUM_HEADS,
                                dropout=config.DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = torch.tensor([4.0], device=DEVICE)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    train_data_list = [d for _, _, d in splits["train"]]

    for epoch in range(epochs):
        model.train()
        loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True)
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            y = getattr(batch, y_attr)
            mask = getattr(batch, mask_attr)
            if mask.sum() == 0:
                continue
            x = _apply_feature_ablation(batch.x, feature_ablation)
            optimizer.zero_grad()
            logits = model(x, batch.edge_index, batch.edge_attr, batch.cent,
                           edge_index_channel=getattr(batch, "edge_index_channel", None))
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], y[mask], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, splits["val"], horizon, mask_attr=mask_attr, y_attr=y_attr,
                                feature_ablation=feature_ablation)
        if verbose and epoch % 10 == 0:
            print(f"  epoch {epoch}: loss={total_loss:.3f} val_f1={val_metrics['f1']:.3f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, splits


@torch.no_grad()
def collect_predictions(model, snapshot_list, horizon, mask_attr=None, y_attr=None, feature_ablation="full"):
    mask_attr = mask_attr or f"mask{horizon}"
    y_attr = y_attr or f"y{horizon}"
    model.eval()
    data_list = [d for _, _, d in snapshot_list]
    region_list = [r for r, _, _ in snapshot_list]
    all_probs, all_true, all_region = [], [], []
    # DataLoader can't collate (Data, str) tuples directly; batch manually per chunk
    for start in range(0, len(data_list), BATCH_SIZE):
        chunk = data_list[start:start + BATCH_SIZE]
        chunk_regions = region_list[start:start + BATCH_SIZE]
        batch = Batch.from_data_list(chunk).to(DEVICE)
        y = getattr(batch, y_attr)
        mask = getattr(batch, mask_attr)
        if mask.sum() == 0:
            continue
        x = _apply_feature_ablation(batch.x, feature_ablation)
        logits = model(x, batch.edge_index, batch.edge_attr, batch.cent,
                           edge_index_channel=getattr(batch, "edge_index_channel", None))
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        true = y[mask].cpu().numpy()
        node_region = np.array(chunk_regions)[batch.batch[mask].cpu().numpy()]
        all_probs.append(probs)
        all_true.append(true)
        all_region += node_region.tolist()
    if not all_probs:
        return np.array([]), np.array([]), []
    return np.concatenate(all_probs), np.concatenate(all_true), all_region


def precision_at_k(y_true, y_prob, k):
    if k <= 0 or len(y_true) == 0:
        return float("nan")
    order = np.argsort(-y_prob)[:k]
    return float(np.mean(np.asarray(y_true)[order]))


def compute_metrics(y_true, y_prob, threshold=0.5, y_pred=None):
    """If `y_pred` is given, it's used directly for precision/recall/F1/
    accuracy instead of thresholding `y_prob` -- needed when predictions come
    from a per-snapshot threshold (e.g. each day's own 80th percentile)
    rather than one global cutoff over the pooled scores."""
    if len(y_true) == 0:
        return {k: float("nan") for k in
                ["precision", "recall", "f1", "auc", "pr_auc", "precision_at_k",
                 "accuracy", "n", "pos_rate"]}
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(np.sum(y_true))
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "n": len(y_true),
        "pos_rate": float(np.mean(y_true)),
        "precision_at_k": precision_at_k(y_true, y_prob, n_pos),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def evaluate(model, snapshot_list, horizon, mask_attr=None, y_attr=None, feature_ablation="full"):
    probs, true, _ = collect_predictions(model, snapshot_list, horizon,
                                          mask_attr=mask_attr, y_attr=y_attr, feature_ablation=feature_ablation)
    return compute_metrics(true, probs)


def evaluate_by_region(model, snapshot_list, horizon, mask_attr=None, y_attr=None, feature_ablation="full"):
    probs, true, regions_arr = collect_predictions(model, snapshot_list, horizon,
                                                     mask_attr=mask_attr, y_attr=y_attr, feature_ablation=feature_ablation)
    out = {"overall": compute_metrics(true, probs)}
    regions_arr = np.array(regions_arr)
    for r in sorted(set(regions_arr.tolist())):
        m = regions_arr == r
        out[r] = compute_metrics(true[m], probs[m])
    return out
