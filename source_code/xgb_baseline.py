"""State-of-the-art non-graph tabular baseline: gradient-boosted trees
(XGBoost) on the same node features as MLP, with no message passing. This
directly tests whether the graph structure adds value beyond a strong
tabular learner, which a plain MLP alone cannot rule out as convincingly.
"""
import numpy as np
import xgboost as xgb

from . import config, train_eval


def _flatten(snapshot_list, horizon, mask_attr):
    xs, ys = [], []
    for _, _, data in snapshot_list:
        mask = getattr(data, mask_attr).numpy()
        if mask.sum() == 0:
            continue
        y = getattr(data, f"y{horizon}").numpy()
        xs.append(data.x.numpy()[mask])
        ys.append(y[mask])
    return np.concatenate(xs), np.concatenate(ys)


def train_and_eval_xgb(regions, horizon, mask_attr=None, seed=42):
    mask_attr = mask_attr or f"mask{horizon}"
    all_snaps = train_eval.load_all_snapshots(regions)
    splits = train_eval.collect_split_snapshots(all_snaps, regions, horizon, mask_attr=mask_attr)

    X_train, y_train = _flatten(splits["train"], horizon, mask_attr)
    X_val, y_val = _flatten(splits["val"], horizon, mask_attr)
    X_test, y_test = _flatten(splits["test"], horizon, mask_attr)

    pos_rate = y_train.mean()
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)

    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", early_stopping_rounds=30,
        random_state=seed, n_jobs=16,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    probs = clf.predict_proba(X_test)[:, 1]
    return train_eval.compute_metrics(y_test, probs)


def multi_seed_xgb(regions, horizon, mask_attr=None, seeds=(42, 7, 123)):
    runs = [train_and_eval_xgb(regions, horizon, mask_attr=mask_attr, seed=s) for s in seeds]
    agg = {}
    for key in ["precision", "recall", "f1", "auc", "pr_auc", "precision_at_k"]:
        vals = np.array([r[key] for r in runs], dtype=float)
        agg[f"{key}_mean"] = float(np.nanmean(vals))
        agg[f"{key}_std"] = float(np.nanstd(vals))
    agg["n"] = runs[0]["n"]
    agg["pos_rate"] = runs[0]["pos_rate"]
    agg["runs"] = runs
    return agg


if __name__ == "__main__":
    print("=== XGBoost, standard task, t+1 ===")
    agg = multi_seed_xgb(config.REGIONS, 1)
    print(f"F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
          f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f}")

    print("=== XGBoost, emergence task, t+1 ===")
    agg = multi_seed_xgb(config.REGIONS, 1, mask_attr="emerge_mask1")
    print(f"F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
          f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f} "
          f"P@k={agg['precision_at_k_mean']:.3f}+-{agg['precision_at_k_std']:.3f}")
