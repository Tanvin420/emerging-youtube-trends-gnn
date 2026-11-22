"""Day-level paired significance testing.

Every reported metric elsewhere in this project pools predictions across
all nodes in all test-period snapshots and treats each node-prediction as
an independent sample. But snapshots are consecutive daily graphs with
heavy day-to-day autocorrelation (the whole reason a persistence baseline
is hard to beat) -- so node-level significance overstates how much
independent evidence there actually is. This module instead computes one
metric value per (region, day) snapshot for each model, then paired-
bootstraps over *days* (resampling whole days with replacement, not
individual nodes) to get a confidence interval and a one-sided p-value
for "model A's day-level metric is higher than model B's."
"""
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch_geometric.data import Batch

from . import train_eval

DEVICE = train_eval.DEVICE


@torch.no_grad()
def collect_per_day_predictions(model, snapshot_list, horizon, mask_attr=None, y_attr=None,
                                 feature_ablation="full"):
    """Like train_eval.collect_predictions, but keeps (region, month) per
    prediction so per-day metrics can be computed. Returns a dict keyed by
    (region, month) -> (probs, true) for that single day's eligible nodes."""
    mask_attr = mask_attr or f"mask{horizon}"
    y_attr = y_attr or f"y{horizon}"
    model.eval()
    out = {}
    for region, month, data in snapshot_list:
        batch = Batch.from_data_list([data]).to(DEVICE)
        y = getattr(batch, y_attr)
        mask = getattr(batch, mask_attr)
        if mask.sum() == 0:
            continue
        x = train_eval._apply_feature_ablation(batch.x, feature_ablation)
        logits = model(x, batch.edge_index, batch.edge_attr, batch.cent,
                       edge_index_channel=getattr(batch, "edge_index_channel", None))
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        true = y[mask].cpu().numpy()
        out[(region, month)] = (probs, true)
    return out


def per_day_f1(per_day_preds, threshold=0.5):
    """{(region, month): f1} for days with at least one positive AND one
    negative example (F1 is undefined/degenerate otherwise)."""
    out = {}
    for key, (probs, true) in per_day_preds.items():
        if len(true) == 0 or true.sum() == 0:
            continue
        pred = (probs >= threshold).astype(int)
        out[key] = f1_score(true, pred, zero_division=0)
    return out


def paired_day_bootstrap(metric_a: dict, metric_b: dict, n_boot=10000, seed=0):
    """Paired bootstrap over the days common to both models. Returns the
    observed mean difference (a - b), a 95% CI, and a one-sided p-value
    for "a's day-level metric is stochastically greater than b's"."""
    common_days = sorted(set(metric_a) & set(metric_b))
    a = np.array([metric_a[d] for d in common_days])
    b = np.array([metric_b[d] for d in common_days])
    diffs = a - b
    observed = float(diffs.mean())

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    # one-sided p-value: fraction of bootstrap resamples where the
    # difference is <= 0, i.e. evidence against "a > b"
    p_value = float(np.mean(boot_means <= 0))
    return {
        "n_days": n, "observed_diff": observed,
        "ci_95_low": float(ci_lo), "ci_95_high": float(ci_hi),
        "p_value_a_gt_b": p_value,
    }
