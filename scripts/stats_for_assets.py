"""Compute aggregate stats across all 773 training wells for README visuals."""
import glob
import os
import numpy as np
import pandas as pd

DATA = r"D:\AI数据分析\kaggle比赛\ROGII - 井筒地质预测\data\train"

rows = []
for f in glob.glob(os.path.join(DATA, "*__horizontal_well.csv")):
    well = os.path.basename(f).split("__")[0]
    df = pd.read_csv(f)
    known = df["TVT_input"].notna()
    eval_mask = ~known
    anchor = df.loc[known, "TVT_input"].iloc[-1]
    tvt_eval = df.loc[eval_mask, "TVT"].to_numpy()
    gr_all = df["GR"]
    gr_eval = df.loc[eval_mask, "GR"]
    md = df["MD"].to_numpy()
    const_rmse = float(np.sqrt(np.mean((tvt_eval - anchor) ** 2))) if len(tvt_eval) else np.nan
    const_sse = float(np.sum((tvt_eval - anchor) ** 2)) if len(tvt_eval) else 0.0
    rows.append(
        {
            "well": well,
            "n_rows": len(df),
            "n_known": int(known.sum()),
            "n_eval": int(eval_mask.sum()),
            "md_len_known": float(md[known.to_numpy()].max() - md[known.to_numpy()].min() + 1),
            "md_len_eval": float(md[eval_mask.to_numpy()].max() - md[eval_mask.to_numpy()].min() + 1),
            "gr_missing_all": float(gr_all.isna().mean()),
            "gr_missing_eval": float(gr_eval.isna().mean()),
            "anchor": float(anchor),
            "tvt_drift": float(np.max(np.abs(tvt_eval - anchor))) if len(tvt_eval) else 0.0,
            "const_rmse": const_rmse,
            "const_sse": const_sse,
        }
    )

st = pd.DataFrame(rows)
total_sse = st["const_sse"].sum()
top10 = st.sort_values("const_sse", ascending=False).head(int(np.ceil(len(st) * 0.1)))
top10_share = top10["const_sse"].sum() / total_sse

print("wells:", len(st))
print("rows total:", int(st["n_rows"].sum()))
print("const RMSE overall (eval rows):", round(float(np.sqrt(total_sse / st["n_eval"].sum())), 3))
print("GR missing overall (rows):", round(float((st.n_rows * st.gr_missing_all).sum() / st.n_rows.sum()), 4))
print("GR missing in eval region (rows):", round(float((st.n_eval * st.gr_missing_eval).sum() / st.n_eval.sum()), 4))
print("eval length ft: median", round(st.md_len_eval.median(), 1), "mean", round(st.md_len_eval.mean(), 1),
      "min", round(st.md_len_eval.min(), 1), "max", round(st.md_len_eval.max(), 1))
print("known length ft: median", round(st.md_len_known.median(), 1), "mean", round(st.md_len_known.mean(), 1))
print("n_eval rows: median", int(st.n_eval.median()), "mean", int(st.n_eval.mean()))
print("TVT drift ft: median", round(st.tvt_drift.median(), 1), "mean", round(st.tvt_drift.mean(), 1),
      "p90", round(st.tvt_drift.quantile(0.9), 1), "max", round(st.tvt_drift.max(), 1))
print("const per-well RMSE: median", round(st.const_rmse.median(), 2),
      "mean", round(st.const_rmse.mean(), 2), "p90", round(st.const_rmse.quantile(0.9), 2),
      "max", round(st.const_rmse.max(), 2))
print("top10 SSE share:", round(top10_share, 4))
print("top10 share of wells:", round(len(top10) / len(st), 4))
print("top5 share:", round(st.sort_values("const_sse", ascending=False).head(int(np.ceil(len(st)*0.05)))["const_sse"].sum()/total_sse, 4))
print("top1pct share:", round(st.sort_values("const_sse", ascending=False).head(int(np.ceil(len(st)*0.01)))["const_sse"].sum()/total_sse, 4))

st.to_csv(r"D:\AI数据分析\kaggle比赛\ROGII - 井筒地质预测\rogii-487th-place-solution\scripts\_well_stats.csv", index=False)
print("saved scripts/_well_stats.csv")
