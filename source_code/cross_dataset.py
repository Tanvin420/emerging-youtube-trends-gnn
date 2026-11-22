"""Cross-dataset validation: evaluate models trained on the primary
2020-2022 dataset (config.REGIONS) zero-shot on external datasets/eras that
were never used for training, model selection, or early stopping -- run
through the exact same feature pipeline (per-snapshot centrality/SIS,
TF-IDF keyword graphs, MiniLM text embeddings) with zero dataset-specific
tuning. Two eras are available (config.EXTERNAL_ERAS), bracketing the
training window on both sides:
  - "2018": the classic 2017-2018 dataset, predates training (see
    prepare_external_dataset.py)
  - "2026": a live, actively-updated dataset, postdates training entirely
    (see prepare_external_dataset_2026.py)

Run order (per era):
    python -m source_code.prepare_external_dataset        # era 2018
    python -m source_code.prepare_external_dataset_2026   # era 2026
    python -m source_code.cross_dataset                   # both eras + summary
"""
import json

import numpy as np
import pandas as pd

from . import (build_dataset, centrality, config, data_prep, embeddings,
               graph_construction, keyword_extraction, persistence_baseline,
               train_eval)

MODELS = ["mlp", "gcn", "gat", "cga-gat", "gat-res", "cga-gat-res", "spcga-gat"]


def _result_paths(era: str, horizon: int):
    # era "2018" keeps its original (pre-multi-era) filenames for
    # backward compatibility with already-published results/paper text.
    suffix = "" if era == "2018" else f"_{era}"
    return {
        "stats": config.RESULTS / f"external_dataset_stats{suffix}.csv",
        "json": config.RESULTS / f"cross_dataset{suffix}_h{horizon}.json",
        "csv": config.RESULTS / f"cross_dataset{suffix}_h{horizon}.csv",
    }


def build_external_cache(era: str, force: bool = False) -> None:
    regions, _, _ = config.EXTERNAL_ERAS[era]
    for region in regions:
        data_prep.build_and_cache(region, force=force)
        keyword_extraction.build_and_cache(region, force=force)
        graph_construction.build_and_cache(region, force=force)
        centrality.build_and_cache(region, force=force)
    # extends the existing embedding cache with only the new regions' texts
    embeddings.build_and_cache(regions=config.REGIONS + regions, force=False)
    for region in regions:
        build_dataset.build_and_cache(region, force=force)


def external_dataset_stats(era: str) -> pd.DataFrame:
    regions, region_names, _ = config.EXTERNAL_ERAS[era]
    rows = []
    for region in regions:
        videos = data_prep.build_and_cache(region)
        graphs = graph_construction.build_and_cache(region)
        n_days = len(graphs)
        avg_edges = np.mean([g.number_of_edges() for g in graphs.values()])
        degrees = [d for g in graphs.values() for _, d in g.degree()]
        rows.append({
            "region": region_names[region],
            "video_instances": len(videos),
            "unique_videos": videos["video_id"].nunique(),
            "days": n_days,
            "avg_edges_per_day": round(float(avg_edges), 1),
            "avg_degree": round(float(np.mean(degrees)), 1) if degrees else 0.0,
        })
    return pd.DataFrame(rows)


def _external_snapshot_list(ext_snaps: dict, regions, horizon: int, mask_attr=None):
    """Flat (region, month, data) list over ALL external snapshots with a
    non-empty label mask for this horizon/task -- no train/val/test date
    split, since none of this data is used for training."""
    mask_attr = mask_attr or f"mask{horizon}"
    out = []
    for region in regions:
        for month, data in sorted(ext_snaps[region].items()):
            mask = getattr(data, mask_attr)
            if mask.sum().item() == 0:
                continue
            out.append((region, month, data))
    return out


def run_cross_dataset_eval(era: str, horizon: int = 1, seeds=(42, 7, 123)) -> dict:
    regions, _, _ = config.EXTERNAL_ERAS[era]
    orig_snaps = train_eval.load_all_snapshots(config.REGIONS)
    ext_snaps = {r: build_dataset.build_and_cache(r) for r in regions}
    ext_list = _external_snapshot_list(ext_snaps, regions, horizon)

    results = {}

    print(f"[era {era}] [persistence baseline] (no training; direct evaluation on external data)")
    results["centrality-ranking"] = persistence_baseline.evaluate_persistence(
        None, None, horizon, snapshot_list=ext_list)
    m = results["centrality-ranking"]["overall"]
    print(f"  F1={m['f1']:.3f} AUC={m['auc']:.3f} PR-AUC={m['pr_auc']:.3f}")

    for name in MODELS:
        runs = []
        for seed in seeds:
            model, _ = train_eval.train_model(
                name, config.REGIONS, horizon, seed=seed, verbose=False, all_snaps=orig_snaps)
            m = train_eval.evaluate(model, ext_list, horizon)
            runs.append(m)
        agg = {"overall": {}}
        for key in ["precision", "recall", "f1", "auc", "pr_auc", "precision_at_k"]:
            vals = np.array([r[key] for r in runs], dtype=float)
            agg["overall"][f"{key}_mean"] = float(np.nanmean(vals))
            agg["overall"][f"{key}_std"] = float(np.nanstd(vals))
        agg["overall"]["n"] = runs[0]["n"]
        agg["overall"]["pos_rate"] = runs[0]["pos_rate"]
        agg["runs"] = runs
        # per-region breakdown using the last-trained seed's model (region
        # breakdown is descriptive, not the headline comparison -- avoids
        # N more full training runs just to average per-region numbers)
        agg.update({r: m2 for r, m2 in
                    train_eval.evaluate_by_region(model, ext_list, horizon).items()
                    if r != "overall"})
        results[name] = agg
        o = agg["overall"]
        print(f"[era {era}] [{name}] F1={o['f1_mean']:.3f}+-{o['f1_std']:.3f} "
              f"AUC={o['auc_mean']:.3f}+-{o['auc_std']:.3f} "
              f"PR-AUC={o['pr_auc_mean']:.3f}+-{o['pr_auc_std']:.3f} ({len(seeds)} seeds)")

    out_path = _result_paths(era, horizon)["json"]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    return results


def summarize(results: dict, horizon: int = 1) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        m = res["overall"]
        f1 = m.get("f1", m.get("f1_mean"))
        f1_std = m.get("f1_std", 0.0)
        auc = m.get("auc", m.get("auc_mean"))
        auc_std = m.get("auc_std", 0.0)
        pr_auc = m.get("pr_auc", m.get("pr_auc_mean"))
        pr_auc_std = m.get("pr_auc_std", 0.0)
        rows.append({
            "model": name, "horizon": horizon,
            "f1": f1, "f1_std": f1_std, "auc": auc, "auc_std": auc_std,
            "pr_auc": pr_auc, "pr_auc_std": pr_auc_std,
            "n": m["n"], "pos_rate": m["pos_rate"],
        })
    return pd.DataFrame(rows)


def run_era(era: str, horizon: int = 1, force: bool = False) -> pd.DataFrame:
    build_external_cache(era, force=force)
    paths = _result_paths(era, horizon)

    stats = external_dataset_stats(era)
    stats.to_csv(paths["stats"], index=False)
    print(f"=== era {era}: dataset stats ===")
    print(stats.to_string(index=False))

    res = run_cross_dataset_eval(era, horizon=horizon)
    df = summarize(res, horizon=horizon)
    df.to_csv(paths["csv"], index=False)
    print(f"=== era {era}: cross-dataset results ===")
    print(df.to_string(index=False))
    return df


def cross_era_summary(horizon: int = 1) -> pd.DataFrame:
    """Connects three points in time for each model: in-distribution
    (2020-2022, the training data itself), the pre-training 2018 era, and
    the post-training 2026 era -- i.e. how does zero-shot generalization
    change as the time gap from training grows, on either side of it."""
    with open(config.RESULTS / f"main_comparison_h{horizon}.json") as f:
        in_dist = json.load(f)
    era_frames = {}
    for era in config.EXTERNAL_ERAS:
        path = _result_paths(era, horizon)["json"]
        if path.exists():
            with open(path) as f:
                era_frames[era] = json.load(f)

    in_dist_name = {"centrality-ranking": "centrality-ranking"}
    rows = []
    all_model_names = set(in_dist) | set().union(*[set(v) for v in era_frames.values()]) if era_frames else set(in_dist)
    for name in all_model_names:
        row = {"model": name}
        if name in in_dist:
            m = in_dist[name]["overall"]
            row["f1_2020_22"] = m.get("f1", m.get("f1_mean"))
            row["auc_2020_22"] = m.get("auc", m.get("auc_mean"))
        for era, frame in era_frames.items():
            if name in frame:
                m = frame[name]["overall"]
                row[f"f1_{era}"] = m.get("f1", m.get("f1_mean"))
                row[f"auc_{era}"] = m.get("auc", m.get("auc_mean"))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    out_path = config.RESULTS / f"cross_era_summary_h{horizon}.csv"
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    run_era("2018", horizon=1)
    run_era("2026", horizon=1)
    summary = cross_era_summary(horizon=1)
    print("=== cross-era summary (2018 -> 2020-22 [training] -> 2026) ===")
    print(summary.to_string(index=False))
