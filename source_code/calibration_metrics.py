"""Probability calibration diagnostics: Brier score and Expected
Calibration Error (ECE), overall and per region. Complements
threshold_calibration.py (which fixes a *decision-threshold* skew) by
checking whether the underlying predicted *probabilities* are
well-calibrated in the first place (e.g. does a 0.7 predicted probability
actually correspond to ~70% empirical frequency of the positive class).
"""
import numpy as np


def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Standard equal-width-bin ECE: sum over bins of
    (bin weight) * |mean predicted prob - empirical positive rate|."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def calibration_report(y_true, y_prob, n_bins=10):
    return {
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "mean_pred_prob": float(np.mean(y_prob)) if len(y_prob) else float("nan"),
        "empirical_pos_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "n": len(y_true),
    }
