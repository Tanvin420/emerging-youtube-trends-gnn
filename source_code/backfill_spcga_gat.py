"""Backfills SP-CGA-GAT (Self-Preserving Centrality-Guided Graph Attention
Network) into the three result files that already have every other model
but were generated before SP-CGA-GAT was implemented:
  - results/main_comparison_h1.json / .csv / regional_comparison_h1.csv
  - results/horizon_experiment.json
  - results/emergence_comparison_h1.json

Only trains "spcga-gat" (single run for the main/horizon tables, matching
how every other row in those two tables was produced; 3 seeds for the
emergence table, matching every learned model there) -- does not retrain or
touch any other model's already-recorded numbers.
"""
import json
import time

from . import config, run_experiments, train_eval

NAME = "spcga-gat"


def backfill_main_comparison(horizon=1):
    path = config.RESULTS / f"main_comparison_h{horizon}.json"
    with open(path) as f:
        results = json.load(f)

    all_snaps = train_eval.load_all_snapshots(config.REGIONS)
    t0 = time.time()
    print(f"[{NAME}] training main comparison, horizon={horizon} ...")
    model, splits = train_eval.train_model(NAME, config.REGIONS, horizon, verbose=True, all_snaps=all_snaps)
    test_metrics = train_eval.evaluate_by_region(model, splits["test"], horizon)
    test_metrics["overall"]["train_time_sec"] = time.time() - t0
    results[NAME] = test_metrics
    m = test_metrics["overall"]
    print(f"[{NAME}] test F1={m['f1']:.3f} AUC={m['auc']:.3f} PR-AUC={m['pr_auc']:.3f}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)

    df = run_experiments.summarize_main_comparison(results, horizon=horizon)
    df.to_csv(config.RESULTS / f"main_comparison_h{horizon}.csv", index=False)
    reg_df = run_experiments.summarize_regional(results)
    reg_df.to_csv(config.RESULTS / "regional_comparison_h1.csv", index=False)
    return results


def backfill_horizon(horizons=config.HORIZONS, seed=config.SEED, name=NAME):
    path = config.RESULTS / "horizon_experiment.json"
    with open(path) as f:
        results = json.load(f)

    results[name] = {}
    all_snaps = train_eval.load_all_snapshots(config.REGIONS)
    for h in horizons:
        model, splits = train_eval.train_model(name, config.REGIONS, h, seed=seed, verbose=False, all_snaps=all_snaps)
        metrics = train_eval.evaluate(model, splits["test"], h)
        results[name][h] = metrics
        print(f"[{name}] t+{h}: F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} n={metrics['n']}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def backfill_emergence(horizon=1, seeds=(42, 7, 123)):
    path = config.RESULTS / f"emergence_comparison_h{horizon}.json"
    with open(path) as f:
        results = json.load(f)

    mask_attr = f"emerge_mask{horizon}"
    t0 = time.time()
    agg = run_experiments._multi_seed_eval(NAME, config.REGIONS, horizon, mask_attr=mask_attr, seeds=seeds)
    agg["train_time_sec"] = time.time() - t0
    results[NAME] = agg
    print(f"[{NAME}] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
          f"AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
          f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f} "
          f"P@k={agg['precision_at_k_mean']:.3f}+-{agg['precision_at_k_std']:.3f} "
          f"({agg['train_time_sec']:.1f}s, {len(seeds)} seeds)")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


if __name__ == "__main__":
    backfill_main_comparison(horizon=1)
    backfill_horizon()
    backfill_emergence(horizon=1)
