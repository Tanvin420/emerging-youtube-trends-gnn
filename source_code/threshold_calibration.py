"""Per-region decision-threshold calibration.

The Discussion in the paper diagnoses a region-specific skew: every
learned model over-predicts the positive class for India specifically
(higher recall, lower precision than USA/UK/Canada), even though each
region's top-20% label cutoff is computed independently per snapshot (so
it is not a label-distribution artifact) and the skew appears even for
the graph-free MLP (so it is not purely a topology effect). The likeliest
cause identified there: a single global decision threshold (0.5 on the
sigmoid output) applied across four regions whose underlying predicted-
probability distributions differ -- and a per-region calibrated threshold
is flagged as "a cheap fix worth trying first," but was never actually
tried. This module tries it.

Calibration: for each region, pick the threshold on the *validation* set
that maximizes region-level F1 (via sklearn's exact precision_recall_curve
sweep, not a coarse grid), then apply that fixed per-region threshold to
the held-out *test* set -- never touching test-set labels during
calibration, so this is a legitimate post-hoc fix, not test-set leakage.
"""
import numpy as np
from sklearn.metrics import precision_recall_curve

from . import train_eval


def _best_threshold(y_true, y_prob):
    """Threshold (on sklearn's precision_recall_curve grid) that maximizes
    F1 for this one region's validation subset. Falls back to 0.5 if the
    region has too few examples or no positives to calibrate against."""
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision/recall have one more entry than thresholds (the last point
    # is recall=0); align by dropping that trailing point.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where((precision + recall) > 0,
                   2 * precision * recall / np.maximum(precision + recall, 1e-12),
                   0.0)
    return float(thresholds[np.argmax(f1)])


def calibrate_regional_thresholds(model, splits, horizon, mask_attr=None, feature_ablation="full"):
    """Returns {region: threshold} calibrated on splits['val']."""
    probs, true, region = train_eval.collect_predictions(
        model, splits["val"], horizon, mask_attr=mask_attr, feature_ablation=feature_ablation)
    region = np.array(region)
    thresholds = {}
    for r in sorted(set(region.tolist())):
        m = region == r
        thresholds[r] = _best_threshold(true[m], probs[m])
    return thresholds


def evaluate_with_calibrated_thresholds(model, splits, horizon, mask_attr=None, feature_ablation="full"):
    """Calibrates per-region thresholds on validation, applies them to the
    test set, and returns both the calibrated and the global-0.5-threshold
    results (overall + per region) for direct comparison."""
    thresholds = calibrate_regional_thresholds(
        model, splits, horizon, mask_attr=mask_attr, feature_ablation=feature_ablation)

    probs, true, region = train_eval.collect_predictions(
        model, splits["test"], horizon, mask_attr=mask_attr, feature_ablation=feature_ablation)
    region = np.array(region)

    y_pred_calibrated = np.zeros_like(true, dtype=int)
    for r, thr in thresholds.items():
        m = region == r
        y_pred_calibrated[m] = (probs[m] >= thr).astype(int)

    out = {"thresholds": thresholds, "global": {}, "calibrated": {}}
    out["global"]["overall"] = train_eval.compute_metrics(true, probs)
    out["calibrated"]["overall"] = train_eval.compute_metrics(true, probs, y_pred=y_pred_calibrated)
    for r in sorted(set(region.tolist())):
        m = region == r
        out["global"][r] = train_eval.compute_metrics(true[m], probs[m])
        out["calibrated"][r] = train_eval.compute_metrics(true[m], probs[m], y_pred=y_pred_calibrated[m])
    return out
