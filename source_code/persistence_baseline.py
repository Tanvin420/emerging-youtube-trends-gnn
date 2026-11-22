"""Step 15, Baseline 1 (+ a second non-learned baseline for the emergence task).

Two score-based, no-learning baselines, both computed purely from centrality
already available at time t. Thresholds are computed **per snapshot** (each
day gets its own cut), matching how the top-20% label itself is defined --
pooling scores across many days before thresholding would compare a day's
score against an unrelated day's cutoff.

- `evaluate_persistence`: score = current SIS, thresholded per day against
  *all* nodes in that day's snapshot -- i.e. it reproduces `current_top`
  exactly. "Structural influence persists." On the *emergence* task this is
  a correct floor, not just a weak baseline: eligible nodes are defined as
  those below this exact cut, so persistence predicts 0 for all of them by
  construction (zero recall) -- it cannot get credit for the transitions the
  task is actually about.

- `evaluate_trend`: score = mean of the 4 delta-centrality features (recent
  structural momentum), thresholded per day against the *eligible*
  population only (there's no pre-existing "currently rising" label to
  reproduce, so we rank the eligible pool each day and take its own top
  20%). This is the sensible non-learned baseline for the emergence task:
  "videos whose centrality is already rising are the ones most likely to
  cross the top-20% line next."
"""
import numpy as np

from . import config, train_eval


def _evaluate_score_baseline(all_snaps, regions, horizon, score_fn, mask_attr=None,
                              threshold_population="all", snapshot_list=None):
    """`snapshot_list`, if given, is evaluated directly instead of deriving a
    test split from `all_snaps`/`regions` via the fixed train/val/test date
    cutoffs -- needed for cross-dataset validation, where the external data
    is entirely out-of-distribution evaluation and never split by those
    (in-distribution-specific) date boundaries."""
    mask_attr = mask_attr or f"mask{horizon}"
    if snapshot_list is None:
        splits = train_eval.collect_split_snapshots(all_snaps, regions, horizon, mask_attr=mask_attr)
        snapshot_list = splits["test"]
    probs, preds, true, regions_arr = [], [], [], []
    for region, month, data in snapshot_list:
        y = getattr(data, f"y{horizon}")
        mask = getattr(data, mask_attr)
        if mask.sum() == 0:
            continue
        mask_np = mask.cpu().numpy()
        score_all = score_fn(data)

        # per-snapshot threshold: over all nodes (reproduces current_top) or
        # over just the eligible/masked nodes (no natural "all nodes" cut)
        pop_scores = score_all if threshold_population == "all" else score_all[mask_np]
        cut = np.quantile(pop_scores, config.TOP_PERCENTILE)

        eligible_scores = score_all[mask_np]
        probs.append(eligible_scores)
        preds.append((eligible_scores >= cut).astype(int))
        true.append(y[mask].cpu().numpy())
        regions_arr += [region] * int(mask.sum().item())
    if not probs:
        return {"overall": train_eval.compute_metrics(np.array([]), np.array([]))}
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    true = np.concatenate(true)
    regions_arr = np.array(regions_arr)

    out = {"overall": train_eval.compute_metrics(true, probs, y_pred=preds)}
    for r in sorted(set(regions_arr.tolist())):
        m_idx = regions_arr == r
        out[r] = train_eval.compute_metrics(true[m_idx], probs[m_idx], y_pred=preds[m_idx])
    return out


def _persistence_score(data):
    return data.sis.cpu().numpy()


def _trend_score(data):
    return data.x[:, config.DELTA_FEATURE_SLICE].mean(dim=1).cpu().numpy()


def evaluate_persistence(all_snaps, regions, horizon, mask_attr=None, snapshot_list=None):
    return _evaluate_score_baseline(all_snaps, regions, horizon, _persistence_score,
                                     mask_attr=mask_attr, threshold_population="all",
                                     snapshot_list=snapshot_list)


def evaluate_trend(all_snaps, regions, horizon, mask_attr=None, snapshot_list=None):
    return _evaluate_score_baseline(all_snaps, regions, horizon, _trend_score,
                                     mask_attr=mask_attr, threshold_population="eligible",
                                     snapshot_list=snapshot_list)
