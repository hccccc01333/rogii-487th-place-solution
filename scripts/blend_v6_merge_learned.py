# -*- coding: utf-8 -*-
"""Full-773 blend test: v6 merge (main_proxy + full soft) x S3 learned branch.

Also saves per-row v6 merge and learned for the v7 kernel decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))
from alignment.pf128_softgate_pipeline import predict_gate, train_gate  # noqa: E402


def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def warmup(md_since, tau=85.0):
    return 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau)


def sg_smooth(values, well_ids, win=61, poly=3):
    out = values.copy()
    for wid in pd.unique(well_ids):
        idx = np.flatnonzero(well_ids == wid)
        v = values[idx]
        wl = min(win, len(v))
        if wl % 2 == 0:
            wl -= 1
        if wl >= poly + 2:
            out[idx] = savgol_filter(v, wl, poly)
    return out


def main() -> None:
    feat = pd.read_parquet(ROOT / "project/_a1_tmp/s3_full773/features_learned_full773.parquet")
    comp = pd.read_parquet(
        ROOT / "project/artifacts/pf128_softgate/assets/components_train.parquet",
        columns=["well_id", "order", "row_idx", "fold_id", "y_true", "cb", "lik5",
                 "a2", "prefix_calib"])
    main_oof = pd.read_parquet(
        ROOT / "project/outputs/ms_align_full/perwell_selector/oof_preds.parquet",
        columns=["well_id", "order", "pred_gated"])
    comp = comp.merge(main_oof, on=["well_id", "order"], how="left")
    m = feat.merge(comp, on=["well_id", "row_idx"], how="left")
    m = m[m["y_true"].notna()].reset_index(drop=True)
    y = m["y_true"].to_numpy(float)

    learned = sg_smooth(
        m["last_known_tvt"].to_numpy(float)
        + 0.60 * warmup(m["md_since"].to_numpy(float)) * m["model_delta"].to_numpy(float)
        + 0.40 * (m["lik5"].to_numpy(float) - m["last_known_tvt"].to_numpy(float)),
        m["well_id"].to_numpy())

    # v6-style full soft (real a2 + real features), fold-aware gate
    well = pd.read_csv(ROOT / "project/artifacts/pf128_softgate/assets/well_features_train.csv")
    soft_full = np.full(len(m), np.nan)
    for f in range(5):
        tr = m["fold_id"] != f
        va = m["fold_id"] == f
        gate = train_gate(comp[tr], well[well["well_id"].isin(comp.loc[tr, "well_id"])])
        well_va = well[well["well_id"].isin(comp.loc[va, "well_id"])]
        c_va = comp[va].copy()
        soft_full[va.to_numpy()] = predict_gate(c_va, well_va, gate, has_a2=True)["pred"].to_numpy(float)
    merge = (0.6 * m["pred_gated"].to_numpy(float) + 0.4 * soft_full).astype(float)
    fold = m["fold_id"].to_numpy()

    rows = pd.DataFrame({"well_id": m["well_id"], "row_idx": m["row_idx"],
                         "fold_id": m["fold_id"], "y_true": y,
                         "merge_v6": merge, "learned": learned})
    rows.to_parquet(ROOT / "project/_a1_tmp/s3_full773/blend_rows.parquet", index=False)

    res = {"n_rows": int(len(m)),
           "merge_v6": round(rmse(y, merge), 4),
           "learned": round(rmse(y, learned), 4)}
    for w in (0.6, 0.5, 0.4, 0.3, 0.2):
        p = w * merge + (1.0 - w) * learned
        res[f"w{w}"] = round(rmse(y, p), 4)
        res[f"w{w}_folds"] = {int(f): round(rmse(y[fold == f], p[fold == f]), 4) for f in range(5)}
    print(json.dumps(res, indent=2))
    (ROOT / "project/outputs/ms_align_full/v6_merge_x_learned_blend.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
