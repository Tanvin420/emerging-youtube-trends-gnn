"""Step 22: qualitative emerging-video case study.

An "emerging" video is one with low/moderate current structural influence
(current SIS below the median) that becomes structurally influential in the
future (top-20% SIS at t+h). We compare each model's predicted probability
for these specific videos to see whether CGA-GAT anticipates them earlier
than the centrality-ranking, MLP, and GAT baselines.
"""
import numpy as np
import pandas as pd
import torch

from . import config, persistence_baseline, train_eval


@torch.no_grad()
def _predict_all(model, snapshot_list, horizon):
    """Returns a DataFrame with one row per (region, month, video_id)."""
    model.eval()
    rows = []
    for region, month, data in snapshot_list:
        mask = getattr(data, f"mask{horizon}")
        y = getattr(data, f"y{horizon}")
        if mask.sum() == 0:
            continue
        d = data.to(train_eval.DEVICE)
        logits = model(d.x, d.edge_index, d.edge_attr, d.cent)
        probs = torch.sigmoid(logits).cpu().numpy()
        sis_now = data.sis.cpu().numpy()
        mask_np = mask.cpu().numpy()
        y_np = y.cpu().numpy()
        vids = np.array(data.video_id)
        for i in np.where(mask_np)[0]:
            rows.append({
                "region": region, "month": month, "video_id": vids[i],
                "current_sis": sis_now[i], "future_top20": int(y_np[i]),
                "pred_prob": probs[i],
            })
    return pd.DataFrame(rows)


def emerging_case_study(models_dict: dict, splits_by_model: dict, horizon=1, split="test"):
    """models_dict: {name: trained_model}; splits_by_model: {name: splits} from train_model.
    Returns a DataFrame of emerging videos (current_sis < median, future_top20 == 1)
    with each model's predicted probability attached as a column.
    """
    merged = None
    for name, model in models_dict.items():
        snaps = splits_by_model[name][split]
        df = _predict_all(model, snaps, horizon)
        df = df.rename(columns={"pred_prob": f"pred_{name}"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df[["region", "month", "video_id", f"pred_{name}"]],
                on=["region", "month", "video_id"], how="left",
            )

    median_sis = merged["current_sis"].median()
    emerging = merged[(merged["current_sis"] < median_sis) & (merged["future_top20"] == 1)].copy()
    return emerging.sort_values("current_sis")


if __name__ == "__main__":
    regions = config.REGIONS
    horizon = 1
    models_dict, splits_by_model = {}, {}
    for name in ["mlp", "gat", "cga-gat"]:
        model, splits = train_eval.train_model(name, regions, horizon)
        models_dict[name] = model
        splits_by_model[name] = splits

    df = emerging_case_study(models_dict, splits_by_model, horizon=horizon)
    df.to_csv(config.RESULTS / "emerging_videos.csv", index=False)
    print(df.describe())
    for name in models_dict:
        print(name, "mean predicted prob on emerging set:", df[f"pred_{name}"].mean())
