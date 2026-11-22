"""Orchestrates the experiments described in the plan:
main comparison, ablation, forecasting-horizon, and regional analysis.
Saves results as JSON/CSV under results/. No fabricated numbers -- every
value here comes from an actual training + evaluation run.
"""
import json
import time

import numpy as np
import pandas as pd

from . import (calibration_metrics, config, persistence_baseline, significance,
               threshold_calibration, train_eval)

MODELS = ["mlp", "gcn", "sage", "gat", "cga-gat", "gat-res", "cga-gat-res", "spcga-gat"]


def run_main_comparison(regions=config.REGIONS, horizon=1):
    all_snaps = train_eval.load_all_snapshots(regions)
    results = {}

    print(f"[persistence baseline] horizon={horizon}")
    results["centrality-ranking"] = persistence_baseline.evaluate_persistence(all_snaps, regions, horizon)

    for name in MODELS:
        t0 = time.time()
        print(f"[{name}] training horizon={horizon} ...")
        model, splits = train_eval.train_model(name, regions, horizon, verbose=True)
        test_metrics = train_eval.evaluate_by_region(model, splits["test"], horizon)
        dt = time.time() - t0
        test_metrics["overall"]["train_time_sec"] = dt
        results[name] = test_metrics
        print(f"[{name}] test F1={test_metrics['overall']['f1']:.3f} "
              f"AUC={test_metrics['overall']['auc']:.3f} ({dt:.1f}s)")

    out_path = config.RESULTS / f"main_comparison_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_horizon_experiment(regions=config.REGIONS,
                            model_names=("centrality-ranking", "mlp", "sage", "gat", "cga-gat", "cga-gat-res", "spcga-gat")):
    """Compares the persistence baseline against learned models across
    forecast horizons -- the main comparison at h=1 found the persistence
    baseline hard to beat, so this checks whether that holds as the horizon
    grows and day-to-day persistence weakens."""
    results = {name: {} for name in model_names}
    all_snaps = train_eval.load_all_snapshots(regions)
    for h in config.HORIZONS:
        if "centrality-ranking" in model_names:
            results["centrality-ranking"][h] = persistence_baseline.evaluate_persistence(all_snaps, regions, h)["overall"]
            print(f"[centrality-ranking] t+{h}: F1={results['centrality-ranking'][h]['f1']:.3f} "
                  f"AUC={results['centrality-ranking'][h]['auc']:.3f}")
        for name in model_names:
            if name == "centrality-ranking":
                continue
            model, splits = train_eval.train_model(name, regions, h, verbose=False)
            metrics = train_eval.evaluate(model, splits["test"], h)
            results[name][h] = metrics
            print(f"[{name}] t+{h}: F1={metrics['f1']:.3f} AUC={metrics['auc']:.3f} n={metrics['n']}")
    out_path = config.RESULTS / "horizon_experiment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def _multi_seed_eval(model_name, regions, horizon, mask_attr=None, y_attr=None, feature_ablation="full",
                      seeds=(42, 7, 123)):
    """Trains `model_name` from scratch for each seed and returns per-metric
    mean/std over the seeds, plus the raw per-seed runs. Use this instead of
    a single training run whenever a reported gap needs to be judged against
    run-to-run variance (GAT-family models were found to have high variance
    on the imbalanced emergence task)."""
    runs = []
    for seed in seeds:
        model, splits = train_eval.train_model(
            model_name, regions, horizon, mask_attr=mask_attr, y_attr=y_attr,
            feature_ablation=feature_ablation, seed=seed, verbose=False,
        )
        m = train_eval.evaluate(model, splits["test"], horizon, mask_attr=mask_attr, y_attr=y_attr,
                                 feature_ablation=feature_ablation)
        runs.append(m)
    agg = {}
    for key in ["precision", "recall", "f1", "auc", "pr_auc", "precision_at_k"]:
        vals = np.array([r[key] for r in runs], dtype=float)
        agg[f"{key}_mean"] = float(np.nanmean(vals))
        agg[f"{key}_std"] = float(np.nanstd(vals))
    agg["n"] = runs[0]["n"]
    agg["pos_rate"] = runs[0]["pos_rate"]
    agg["runs"] = runs
    return agg


def run_centrality_ablation(regions=config.REGIONS, horizon=1, mask_attr=None, seeds=(42, 7, 123)):
    """2x2 ablation isolating whether CGA-GAT's attention bias adds anything
    beyond having centrality available as a plain node feature (which every
    model, including plain GAT, already receives). Without this, "GAT vs
    CGA-GAT" is confounded: both see centrality in x.

                        attn bias OFF   attn bias ON
    centrality in x ON   gat             cga-gat
    centrality in x OFF  gat_nocent      cga-gat_attnonly

    Multiple seeds so the (previously found to be small) GAT/CGA-GAT gap
    can be judged against run-to-run variance rather than a single draw.
    """
    variants = [
        ("gat", "full"),
        ("gat_nocent", "no_centrality"),
        ("cga-gat", "full"),
        ("cga-gat_attnonly", "no_centrality"),
    ]
    results = {}
    for label, ablation in variants:
        model_name = "gat" if label.startswith("gat") else "cga-gat"
        agg = _multi_seed_eval(model_name, regions, horizon, mask_attr=mask_attr,
                                feature_ablation=ablation, seeds=seeds)
        results[label] = agg
        print(f"[{label}] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
              f"AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
              f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f} "
              f"(n={len(seeds)} seeds)")

    tag = "emergence" if mask_attr else "full"
    out_path = config.RESULTS / f"centrality_ablation_{tag}_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_graph_randomization_check(regions=config.REGIONS, horizon=1, mask_attr=None,
                                   model_names=("gcn", "gat", "sage"), seeds=(42, 7, 123)):
    """Graph-structure sanity check: train on the real keyword-overlap graph
    vs. a degree-preserving edge-shuffled version of it (node features,
    including real centrality, are identical in both -- only which nodes are
    *connected* changes). If real clearly beats shuffled, message passing is
    using genuine topology, not just centrality-as-feature. If they're close,
    it isn't.
    """
    from . import graph_randomization

    real_snaps = train_eval.load_all_snapshots(regions)
    results = {}
    for name in model_names:
        real_runs, rand_runs = [], []
        for seed in seeds:
            model, splits = train_eval.train_model(name, regions, horizon, mask_attr=mask_attr,
                                                     seed=seed, all_snaps=real_snaps, verbose=False)
            real_runs.append(train_eval.evaluate(model, splits["test"], horizon, mask_attr=mask_attr))

            rand_snaps = graph_randomization.randomize_all_snapshots(real_snaps, seed=seed)
            model_r, splits_r = train_eval.train_model(name, regions, horizon, mask_attr=mask_attr,
                                                         seed=seed, all_snaps=rand_snaps, verbose=False)
            rand_runs.append(train_eval.evaluate(model_r, splits_r["test"], horizon, mask_attr=mask_attr))

        def _agg(runs):
            agg = {}
            for key in ["f1", "auc", "pr_auc", "precision_at_k"]:
                vals = np.array([r[key] for r in runs], dtype=float)
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
            agg["runs"] = runs
            return agg

        results[name] = {"real": _agg(real_runs), "randomized": _agg(rand_runs)}
        r, s = results[name]["real"], results[name]["randomized"]
        print(f"[{name}] real F1={r['f1_mean']:.3f}+-{r['f1_std']:.3f} AUC={r['auc_mean']:.3f}+-{r['auc_std']:.3f} | "
              f"random F1={s['f1_mean']:.3f}+-{s['f1_std']:.3f} AUC={s['auc_mean']:.3f}+-{s['auc_std']:.3f}")

    tag = "emergence" if mask_attr else "full"
    out_path = config.RESULTS / f"graph_randomization_{tag}_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_feature_ablation_hierarchy(regions=config.REGIONS, horizon=1, mask_attr=None,
                                    model_name="mlp", seeds=(42, 7, 123)):
    """Answers "where does the predictive power actually come from?" by
    training `model_name` (MLP by default -- graph structure held out of
    the question here) with successive feature blocks removed or isolated.
    `no_graph_derived` removes centrality+delta, leaving only text/views/
    category -- i.e. features that don't depend on the graph at all.
    `centrality_only` is the reviewer-requested "how much is just centrality
    persistence" check.
    """
    order = ["full", "no_text", "no_views", "no_category", "no_centrality",
             "no_delta", "no_graph_derived", "centrality_only", "text_only"]
    results = {}
    for name in order:
        agg = _multi_seed_eval(model_name, regions, horizon, mask_attr=mask_attr,
                                feature_ablation=name, seeds=seeds)
        results[name] = agg
        print(f"[{name}] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
              f"AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
              f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f}")

    tag = "emergence" if mask_attr else "full"
    out_path = config.RESULTS / f"feature_ablation_{tag}_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_emergence_comparison(regions=config.REGIONS, horizon=1,
                              model_names=("mlp", "gcn", "sage", "gat", "cga-gat", "gat-res", "cga-gat-res", "spcga-gat"),
                              seeds=(42, 7, 123)):
    """The reframed primary task: predict which currently-NOT-top-20% videos
    become top-20% at t+h (see build_dataset.py's emerge_mask{h}). Unlike
    the raw top-20%-status task, the persistence baseline cannot get any
    recall here by construction, so it's included only as an explicit floor;
    the "trend" baseline (rank by recent centrality momentum) is the
    sensible non-learned comparison point. Learned models use multiple seeds
    -- GAT/CGA-GAT were found to be far less stable than MLP/GCN on this
    harder, more imbalanced (~4% positive) task, so a single run would be
    misleading either way it happened to land.
    """
    mask_attr = f"emerge_mask{horizon}"
    all_snaps = train_eval.load_all_snapshots(regions)
    results = {}

    results["persistence-floor"] = persistence_baseline.evaluate_persistence(
        all_snaps, regions, horizon, mask_attr=mask_attr)
    results["trend-momentum"] = persistence_baseline.evaluate_trend(
        all_snaps, regions, horizon, mask_attr=mask_attr)
    for name in ["persistence-floor", "trend-momentum"]:
        m = results[name]["overall"]
        print(f"[{name}] F1={m['f1']:.3f} AUC={m['auc']:.3f} PR-AUC={m['pr_auc']:.3f} "
              f"P@k={m['precision_at_k']:.3f} pos_rate={m['pos_rate']:.4f}")

    for name in model_names:
        t0 = time.time()
        agg = _multi_seed_eval(name, regions, horizon, mask_attr=mask_attr, seeds=seeds)
        agg["train_time_sec"] = time.time() - t0
        results[name] = agg
        print(f"[{name}] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
              f"AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
              f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f} "
              f"P@k={agg['precision_at_k_mean']:.3f}+-{agg['precision_at_k_std']:.3f} "
              f"({agg['train_time_sec']:.1f}s, {len(seeds)} seeds)")

    out_path = config.RESULTS / f"emergence_comparison_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


CHANNEL_MODELS = ["spcga-gat", "dual-spcga-gat", "spcga-gat-channelonly"]


def run_channel_augmentation_experiment(regions=config.REGIONS, horizon=1, seeds=(42, 7, 123)):
    """Tests whether a second, structurally distinct edge type -- same-
    channel co-trending videos (channel_graph.py), invisible to the
    keyword-overlap graph and therefore not automatically redundant with
    centrality the way CGA-GAT's attention bias turned out to be (see
    run_centrality_ablation) -- adds anything beyond the keyword graph
    alone. Three arms, all SP-CGA-GAT, differing only in which edge type(s)
    DualSPCGAGATConv aggregates over:
      spcga-gat              keyword edges only (existing best model, as a
                              same-seeds/same-code-path baseline for this
                              specific comparison -- not merely reusing the
                              main_comparison_h1.json numbers)
      dual-spcga-gat          keyword + channel edges together
      spcga-gat-channelonly  channel edges only (isolates the channel
                              signal's standalone value, mirroring
                              run_centrality_ablation's deconfounded style)
    Both the raw status task and the harder emergence task are run, with
    the same 3 seeds used throughout the project.
    """
    results = {"status": {}, "emergence": {}}
    for task, mask_attr in [("status", f"mask{horizon}"), ("emergence", f"emerge_mask{horizon}")]:
        for name in CHANNEL_MODELS:
            t0 = time.time()
            agg = _multi_seed_eval(name, regions, horizon, mask_attr=mask_attr, seeds=seeds)
            agg["train_time_sec"] = time.time() - t0
            results[task][name] = agg
            print(f"[{task}/{name}] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
                  f"AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
                  f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f} "
                  f"({agg['train_time_sec']:.1f}s, {len(seeds)} seeds)")

    out_path = config.RESULTS / f"channel_augmentation_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_regional_calibration_experiment(regions=config.REGIONS, horizon=1,
                                         model_name="spcga-gat", seeds=(42, 7, 123)):
    """Tests the fix the Discussion flags but never tries: replacing the
    single global 0.5 decision threshold with a per-region threshold
    calibrated on the validation set (threshold_calibration.py), for the
    India over-prediction skew found in every learned model. Trains
    `model_name` fresh per seed (so this is directly comparable to the
    main-comparison numbers, not reusing cached predictions), calibrates
    on that seed's validation split, and reports global-vs-calibrated
    metrics -- overall and per region -- averaged over seeds. Run on both
    the raw status task and the harder emergence task, since the skew was
    documented on both.
    """
    results = {"status": {}, "emergence": {}}
    for task, mask_attr in [("status", f"mask{horizon}"), ("emergence", f"emerge_mask{horizon}")]:
        global_runs, calibrated_runs = [], []
        per_region_global = {r: [] for r in regions}
        per_region_calibrated = {r: [] for r in regions}
        thresholds_by_seed = []
        for seed in seeds:
            model, splits = train_eval.train_model(
                model_name, regions, horizon, mask_attr=mask_attr, seed=seed, verbose=False)
            cal = threshold_calibration.evaluate_with_calibrated_thresholds(
                model, splits, horizon, mask_attr=mask_attr)
            global_runs.append(cal["global"]["overall"])
            calibrated_runs.append(cal["calibrated"]["overall"])
            thresholds_by_seed.append(cal["thresholds"])
            for r in regions:
                if r in cal["global"]:
                    per_region_global[r].append(cal["global"][r])
                    per_region_calibrated[r].append(cal["calibrated"][r])

        def _agg(runs):
            agg = {}
            for key in ["precision", "recall", "f1", "auc", "pr_auc"]:
                vals = np.array([r[key] for r in runs], dtype=float)
                agg[f"{key}_mean"] = float(np.nanmean(vals))
                agg[f"{key}_std"] = float(np.nanstd(vals))
            return agg

        task_result = {
            "global_overall": _agg(global_runs),
            "calibrated_overall": _agg(calibrated_runs),
            "global_by_region": {r: _agg(v) for r, v in per_region_global.items() if v},
            "calibrated_by_region": {r: _agg(v) for r, v in per_region_calibrated.items() if v},
            "thresholds_by_seed": thresholds_by_seed,
        }
        results[task] = task_result

        g, c = task_result["global_overall"], task_result["calibrated_overall"]
        print(f"[{task}] global    F1={g['f1_mean']:.3f}+-{g['f1_std']:.3f} "
              f"prec={g['precision_mean']:.3f} rec={g['recall_mean']:.3f}")
        print(f"[{task}] calibrated F1={c['f1_mean']:.3f}+-{c['f1_std']:.3f} "
              f"prec={c['precision_mean']:.3f} rec={c['recall_mean']:.3f}")
        for r in regions:
            gr = task_result["global_by_region"].get(r)
            cr = task_result["calibrated_by_region"].get(r)
            if gr and cr:
                print(f"  [{task}/{r}] global prec={gr['precision_mean']:.3f} rec={gr['recall_mean']:.3f} "
                      f"-> calibrated prec={cr['precision_mean']:.3f} rec={cr['recall_mean']:.3f}")

    out_path = config.RESULTS / f"regional_calibration_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_percountry_comparison(regions=config.REGIONS, horizon=1, model_name="spcga-gat", seeds=(42, 7, 123)):
    """Tests whether a single pooled model (trained on all four regions
    together, the default throughout this project) hides country-level
    heterogeneity that separate, region-specific models would not have.
    The pooled arm reuses results/regional_calibration_h{horizon}.json's
    already-computed per-region breakdown for `model_name` (no retraining)
    rather than duplicating that run; the per-country arm trains
    `model_name` from scratch on each single region (so on 1/4 the data),
    evaluated on that region's own held-out test split.
    """
    pooled_path = config.RESULTS / f"regional_calibration_h{horizon}.json"
    with open(pooled_path) as f:
        pooled = json.load(f)["status"]["global_by_region"]

    per_country = {}
    for r in regions:
        agg = _multi_seed_eval(model_name, [r], horizon, seeds=seeds)
        per_country[r] = agg
        print(f"[per-country/{r}] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
              f"prec={agg['precision_mean']:.3f} rec={agg['recall_mean']:.3f} "
              f"(pooled: F1={pooled[r]['f1_mean']:.3f}, "
              f"prec={pooled[r]['precision_mean']:.3f}, rec={pooled[r]['recall_mean']:.3f})")

    results = {"pooled_by_region": pooled, "per_country": per_country}
    out_path = config.RESULTS / f"percountry_comparison_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_selfpreservation_ladder(regions=config.REGIONS, horizon=1, seeds=(42, 7, 123)):
    """The one rung missing from the existing model comparisons: SP-GAT
    (models.SPCGAGATModel with use_centrality=False) -- self-preserving
    attention with NO centrality-guided bias at all, vs. SP-CGA-GAT
    (same architecture, centrality bias on). Completes a clean ladder
    isolating self-preservation from centrality-guided attention:
        GAT (self-loop, no centrality bias)
        -> CGA-GAT (self-loop, + centrality bias)          [redundant, Table II]
        -> SP-GAT (dedicated self-transform, no bias)       [this function]
        -> SP-CGA-GAT (dedicated self-transform, + bias)    [main result]
        -> GraphSAGE (dedicated self-transform, no attention at all)
    on both task framings, 3 seeds.
    """
    results = {"status": {}, "emergence": {}}
    for task, mask_attr in [("status", f"mask{horizon}"), ("emergence", f"emerge_mask{horizon}")]:
        agg = _multi_seed_eval("sp-gat", regions, horizon, mask_attr=mask_attr, seeds=seeds)
        results[task]["sp-gat"] = agg
        print(f"[{task}/sp-gat] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
              f"AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
              f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f}")
    out_path = config.RESULTS / f"selfpreservation_ladder_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_survivorship_correction_experiment(regions=config.REGIONS, horizon=1, seeds=(42, 7, 123)):
    """Tests a reviewer-flagged survivorship bias: the emergence task's
    eligible population (build_dataset.py's emerge_mask{h}) requires the
    video to still be PRESENT at t+h, silently excluding videos that stop
    trending -- so the task actually being solved is "among videos that
    survive to t+h, which become influential", not "which videos emerge",
    and the excluded population is large (23.5%/43.3%/61.4% of the eligible
    population at t+1/t+2/t+3 disappears rather than staying low or
    becoming high). emerge_mask_all{h} is the corrected eligible population
    -- same "not already top-20% at t" definition, but no longer requiring
    future presence, so a disappeared video is scored as a
    correct-by-construction negative ("did not emerge") instead of being
    dropped. Compares MLP and SP-CGA-GAT under both.
    """
    results = {}
    for name in ["mlp", "spcga-gat"]:
        results[name] = {}
        for variant, mask_attr in [("original_survivors_only", f"emerge_mask{horizon}"),
                                    ("corrected_includes_disappeared", f"emerge_mask_all{horizon}")]:
            agg = _multi_seed_eval(name, regions, horizon, mask_attr=mask_attr, seeds=seeds)
            results[name][variant] = agg
            print(f"[{name}/{variant}] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
                  f"AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
                  f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f} "
                  f"pos_rate={agg['pos_rate']:.4f} n={agg['n']}")
    out_path = config.RESULTS / f"survivorship_correction_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_view_popularity_experiment(regions=config.REGIONS, horizon=1, seeds=(42, 7, 123)):
    """Tests a reviewer-flagged circularity concern: the paper's main label
    (top-20% SIS) is the mean of the same four centralities given to the
    model as input features, so "predicts future top-20% status" risks
    just meaning "predicts future centrality from current centrality."
    view_labels.py defines an independent target sharing no construction
    with centrality: is the video's view count in the top 20% at t+h?
    Trains MLP with different feature blocks visible (full / centrality
    only / no centrality, i.e. views+text+category only) plus SP-CGA-GAT,
    on this view-based label, to see how much of the predictive power
    survives when the target is genuine popularity rather than
    graph-centrality.
    """
    mask_attr = f"mask{horizon}"
    y_attr = f"view_y{horizon}"
    results = {}
    for name, ablation in [("mlp_full", ("mlp", "full")),
                            ("mlp_centrality_only", ("mlp", "centrality_only")),
                            ("mlp_no_graph_derived", ("mlp", "no_graph_derived")),
                            ("spcga-gat", ("spcga-gat", "full"))]:
        model_name, feature_ablation = ablation
        agg = _multi_seed_eval(model_name, regions, horizon, mask_attr=mask_attr, y_attr=y_attr,
                                feature_ablation=feature_ablation, seeds=seeds)
        results[name] = agg
        print(f"[{name}] F1={agg['f1_mean']:.3f}+-{agg['f1_std']:.3f} "
              f"AUC={agg['auc_mean']:.3f}+-{agg['auc_std']:.3f} "
              f"PR-AUC={agg['pr_auc_mean']:.3f}+-{agg['pr_auc_std']:.3f} "
              f"pos_rate={agg['pos_rate']:.4f}")
    out_path = config.RESULTS / f"view_popularity_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_calibration_diagnostics(regions=config.REGIONS, horizon=1,
                                 model_name="spcga-gat", seeds=(42, 7, 123)):
    """Tests a reviewer-flagged gap: every reported metric so far is
    threshold-based (precision/recall/F1) or rank-based (AUC/PR-AUC), but
    neither checks whether the predicted *probabilities* themselves are
    trustworthy -- e.g. whether a 0.7 output actually corresponds to ~70%
    empirical positive frequency. calibration_metrics.py computes Brier
    score and Expected Calibration Error (ECE) directly from each seed's
    raw test-set probabilities (not from the aggregated JSONs, which never
    stored them), overall and per region, on the status task.
    """
    all_snaps = train_eval.load_all_snapshots(regions)
    mask_attr = f"mask{horizon}"
    per_seed = []
    per_region_runs = {r: [] for r in regions}
    for seed in seeds:
        model, splits = train_eval.train_model(
            model_name, regions, horizon, mask_attr=mask_attr, seed=seed, verbose=False,
            all_snaps=all_snaps)
        probs, true, regions_arr = train_eval.collect_predictions(model, splits["test"], horizon,
                                                                    mask_attr=mask_attr)
        per_seed.append(calibration_metrics.calibration_report(true, probs))
        regions_arr = np.array(regions_arr)
        for r in regions:
            m = regions_arr == r
            if m.sum() > 0:
                per_region_runs[r].append(calibration_metrics.calibration_report(true[m], probs[m]))

    def _agg(runs):
        agg = {}
        for key in ["brier", "ece", "mean_pred_prob", "empirical_pos_rate"]:
            vals = np.array([r[key] for r in runs], dtype=float)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        agg["n"] = runs[0]["n"]
        return agg

    overall = _agg(per_seed)
    by_region = {r: _agg(v) for r, v in per_region_runs.items() if v}
    print(f"[calibration/overall] Brier={overall['brier_mean']:.4f}+-{overall['brier_std']:.4f} "
          f"ECE={overall['ece_mean']:.4f}+-{overall['ece_std']:.4f} "
          f"mean_pred={overall['mean_pred_prob_mean']:.4f} empirical_pos={overall['empirical_pos_rate_mean']:.4f}")
    for r, br in by_region.items():
        print(f"  [calibration/{r}] Brier={br['brier_mean']:.4f} ECE={br['ece_mean']:.4f} "
              f"mean_pred={br['mean_pred_prob_mean']:.4f} empirical_pos={br['empirical_pos_rate_mean']:.4f}")

    results = {"model": model_name, "overall": overall, "by_region": by_region}
    out_path = config.RESULTS / f"calibration_diagnostics_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def run_significance_test(regions=config.REGIONS, horizon=1,
                           model_a="spcga-gat", model_b="sage", seeds=(42, 7, 123)):
    """Tests a reviewer-flagged gap: every F1/AUC number in the project
    pools predictions across all test-period node-snapshots as if each
    node-prediction were an independent sample, but consecutive daily
    snapshots are heavily autocorrelated (the whole reason the persistence
    baseline is competitive), so node-level significance overstates the
    independent evidence behind any reported gap. This instead computes one
    F1 per (region, day) for each model and paired-bootstraps the
    day-level difference (significance.py) -- applied to the closest
    headline comparison in the paper, SP-CGA-GAT vs GraphSAGE on the
    emergence task, across the same 3 seeds used elsewhere (each seed's
    pair of models trained independently, then pooled into one paired
    bootstrap over region-day keys common to both).
    """
    mask_attr = f"emerge_mask{horizon}"
    all_a, all_b = {}, {}
    for seed in seeds:
        model_a_trained, splits = train_eval.train_model(
            model_a, regions, horizon, mask_attr=mask_attr, seed=seed, verbose=False)
        model_b_trained, _ = train_eval.train_model(
            model_b, regions, horizon, mask_attr=mask_attr, seed=seed, verbose=False)
        preds_a = significance.collect_per_day_predictions(model_a_trained, splits["test"], horizon,
                                                             mask_attr=mask_attr)
        preds_b = significance.collect_per_day_predictions(model_b_trained, splits["test"], horizon,
                                                             mask_attr=mask_attr)
        for k, v in preds_a.items():
            all_a[(seed, *k)] = v
        for k, v in preds_b.items():
            all_b[(seed, *k)] = v

    f1_a = significance.per_day_f1(all_a)
    f1_b = significance.per_day_f1(all_b)
    boot = significance.paired_day_bootstrap(f1_a, f1_b)
    print(f"[significance] {model_a} vs {model_b} on emergence, day-level F1, "
          f"{len(seeds)} seeds pooled: n_days={boot['n_days']} "
          f"observed_diff={boot['observed_diff']:.4f} "
          f"95% CI=[{boot['ci_95_low']:.4f}, {boot['ci_95_high']:.4f}] "
          f"p(diff<=0)={boot['p_value_a_gt_b']:.4f}")

    results = {"model_a": model_a, "model_b": model_b, "task": "emergence", "horizon": horizon,
               "seeds": list(seeds), **boot}
    out_path = config.RESULTS / f"significance_h{horizon}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def summarize_main_comparison(results: dict, horizon=1) -> pd.DataFrame:
    """Tolerant of both result shapes present in main_comparison_h{h}.json:
    a single-run dict (precision/recall/f1/auc/accuracy directly under
    "overall") or a multi-seed aggregate (the same keys suffixed "_mean",
    e.g. the xgboost entry) -- falls back to NaN for accuracy, which only
    the single-run shape reports."""
    rows = []
    for name, res in results.items():
        m = res["overall"]

        def _get(key, default=float("nan")):
            if key in m:
                return m[key]
            return m.get(f"{key}_mean", default)

        rows.append({
            "model": name, "horizon": horizon,
            "precision": _get("precision"), "recall": _get("recall"),
            "f1": _get("f1"), "auc": _get("auc"), "accuracy": _get("accuracy"),
            "n": m["n"], "pos_rate": m["pos_rate"],
        })
    return pd.DataFrame(rows)


def summarize_regional(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        for region in config.REGIONS:
            if region not in res:
                continue
            m = res[region]
            rows.append({"model": name, "region": config.REGION_NAMES[region],
                         "f1": m["f1"], "auc": m["auc"], "precision": m["precision"],
                         "recall": m["recall"], "n": m["n"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run_main_comparison(horizon=1)
    df = summarize_main_comparison(res)
    print(df.to_string(index=False))
    df.to_csv(config.RESULTS / "main_comparison_h1.csv", index=False)

    reg_df = summarize_regional(res)
    reg_df.to_csv(config.RESULTS / "regional_comparison_h1.csv", index=False)
    print(reg_df.to_string(index=False))
