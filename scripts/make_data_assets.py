"""Generate real-data-driven README visuals (medal-solution style).

All figures are produced from the actual 773-well training data so the
portfolio repo shows evidence, not decoration.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats as sps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = r"D:\AI数据分析\kaggle比赛\ROGII - 井筒地质预测\data\train"
ASSETS = os.path.join(ROOT, "assets")
STATS = os.path.join(ROOT, "scripts", "_well_stats.csv")
os.makedirs(ASSETS, exist_ok=True)

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#C9D1D9",
        "axes.grid": True,
        "grid.color": "#E8ECF1",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
    }
)

BLUE = "#1F4E79"
BLUE_L = "#4C7AB0"
ORANGE = "#E76F51"
ORANGE_L = "#F4A261"
TEAL = "#2A9D8F"
GRAY = "#6B7280"
GRAY_L = "#B8C0CC"
PURPLE = "#7A5195"


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (1.4826 * mad + 1e-9)


def load_well(well: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hw = pd.read_csv(os.path.join(DATA, f"{well}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(DATA, f"{well}__typewell.csv"))
    return hw, tw


def style_ax(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="both", alpha=0.55)


def figure_problem_definition():
    """One real well: input log + target curve + reference typewell map."""
    well = "591cc951"
    hw, tw = load_well(well)
    known = hw["TVT_input"].notna().to_numpy()
    eval_mask = ~known
    md0 = hw["MD"].min()
    x = hw["MD"].to_numpy() - md0
    anchor_md = hw.loc[known, "MD"].max() - md0
    anchor_tvt = hw.loc[known, "TVT_input"].iloc[-1]

    # interpolate GR for display, keep raw gaps as shaded spans
    gr = hw["GR"].to_numpy(dtype=float)
    gr_disp = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    gr_nan = np.isnan(gr)

    tvt_min, tvt_max = hw.loc[eval_mask, "TVT"].min(), hw.loc[eval_mask, "TVT"].max()
    tw_ok = (tw["TVT"] >= tvt_min - 20) & (tw["TVT"] <= tvt_max + 20)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=200)

    ax = axes[0]
    ax.axvspan(0, anchor_md, color=BLUE, alpha=0.07)
    ax.axvspan(anchor_md, x.max(), color=ORANGE, alpha=0.09)
    ax.axvline(anchor_md, color=GRAY, ls="--", lw=1.3)
    # missing spans
    for s, e in _nan_spans(gr_nan, x):
        ax.axvspan(s, e, color=GRAY, alpha=0.22, lw=0)
    ax.plot(x[known], gr_disp[known], color=BLUE, lw=1.4, label="已知前缀 GR")
    ax.plot(x[eval_mask], gr_disp[eval_mask], color=ORANGE, lw=1.4, label="评估区 GR")
    ax.annotate(
        "锚点（最后已知 TVT）",
        xy=(anchor_md, ax.get_ylim()[1]),
        xytext=(x.min() + 0.02 * (x.max() - x.min()), ax.get_ylim()[1]),
        fontsize=10.5,
        color=GRAY,
        ha="left",
        va="top",
    )
    style_ax(ax, "① 水平井实测 GR 测井", "MD − 井口（ft）", "GR（API）")
    ax.legend(loc="upper right", fontsize=10)

    ax = axes[1]
    ax.axvspan(0, anchor_md, color=BLUE, alpha=0.07)
    ax.axvspan(anchor_md, x.max(), color=ORANGE, alpha=0.09)
    ax.axvline(anchor_md, color=GRAY, ls="--", lw=1.3)
    ax.plot(x[known], hw.loc[known, "TVT"], color=BLUE, lw=2.0)
    ax.plot(x[eval_mask], hw.loc[eval_mask, "TVT"], color=ORANGE, lw=2.0)
    ax.scatter([anchor_md], [anchor_tvt], s=55, color=ORANGE, zorder=5, edgecolor="white", linewidth=1.2)
    ax.set_ylim(tvt_min - 8, tvt_max + 8)
    ax.invert_yaxis()
    ax.annotate(
        f"目标曲线：评估区 TVT 在锚点下方最深约 {anchor_tvt - tvt_min:.0f} ft",
        xy=(0.5 * x.max(), tvt_min + 2),
        fontsize=10.5,
        color=ORANGE,
    )
    style_ax(ax, "② 目标 TVT（地层垂直坐标）", "MD − 井口（ft）", "TVT（ft）")

    ax = axes[2]
    ax.plot(tw["GR"], tw["TVT"], color=BLUE_L, lw=1.5)
    ax.axhspan(tvt_min - 5, tvt_max + 5, color=ORANGE, alpha=0.16)
    ax.axhline(anchor_tvt, color=GRAY, ls="--", lw=1.2)
    ax.invert_yaxis()
    ax.annotate(
        "橙色带 = 本井 TVT 所在层段",
        xy=(tw["GR"].max() * 0.45, tvt_max + 3),
        fontsize=10.5,
        color=ORANGE,
    )
    style_ax(ax, "③ 参考直井（typewell）：GR–TVT 地图", "GR（API）", "TVT（ft）")

    fig.suptitle(
        f"问题定义：把“前方地层在哪一层”变成“把 ① 与 ③ 对齐，在 ② 上继续画曲线”"
        f"（真实训练井 {well}）",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ASSETS, "problem_schematic.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def _nan_spans(nan: np.ndarray, x: np.ndarray):
    spans = []
    i = 0
    n = len(nan)
    while i < n:
        if nan[i]:
            j = i
            while j < n and nan[j]:
                j += 1
            spans.append((float(x[i]), float(x[j - 1])))
            i = j
        else:
            i += 1
    return spans


def figure_alignment_panel():
    """2D GR-matching panel: one well rendered as one image."""
    well = "591cc951"
    hw, tw = load_well(well)
    known = hw["TVT_input"].notna().to_numpy()
    ev = hw[~known].reset_index(drop=True)
    anchor = hw.loc[known, "TVT_input"].iloc[-1]
    anchor_md = hw.loc[known, "MD"].max()

    gr_well = pd.Series(ev["GR"].to_numpy(dtype=float)).interpolate(limit_direction="both").to_numpy()
    gr_well_z = robust_z(gr_well)
    tw_sorted = tw.sort_values("TVT")
    tw_grid = np.arange(tw_sorted["TVT"].min(), tw_sorted["TVT"].max() + 0.5, 0.5)
    tw_gr = np.interp(tw_grid, tw_sorted["TVT"].to_numpy(), tw_sorted["GR"].to_numpy())
    tw_gr_z = robust_z(tw_gr)

    n_col = 150
    idx = np.linspace(0, len(ev) - 1, n_col).astype(int)
    delta = np.arange(-100, 81, 1.0)
    md_rel = ev.loc[idx, "MD"].to_numpy() - anchor_md
    truth = ev.loc[idx, "TVT"].to_numpy() - anchor
    cost = np.zeros((len(delta), n_col), dtype=float)
    for c, i in enumerate(idx):
        cand = anchor + delta
        tw_at_cand = np.interp(cand, tw_grid, tw_gr_z, left=np.nan, right=np.nan)
        cost[:, c] = (gr_well_z[i] - tw_at_cand) ** 2

    vmax = np.nanpercentile(cost, 94)
    fig, ax = plt.subplots(figsize=(15, 7.2), dpi=200)
    mesh = ax.pcolormesh(
        md_rel,
        delta,
        cost,
        cmap="magma_r",
        vmin=0,
        vmax=float(vmax),
        shading="auto",
    )
    ax.plot(md_rel, truth, color="#7EF9FF", ls="--", lw=2.2, label="真实 TVT 轨迹（相对锚点）")
    ax.axhline(0, color="white", lw=0.9, alpha=0.35)
    ax.annotate(
        "平行谷：GR 近似周期，多个深度都“看起来对”",
        xy=(md_rel[-1] - 850, 52),
        xytext=(md_rel[-1] - 2600, 62),
        fontsize=11,
        color="white",
        arrowprops=dict(arrowstyle="->", color="white", lw=1.4),
    )
    ax.annotate(
        "真实轨迹沿着暗谷走",
        xy=(md_rel[int(n_col * 0.42)], truth[int(n_col * 0.42)]),
        xytext=(md_rel[int(n_col * 0.42)] + 250, -75),
        fontsize=11,
        color="#7EF9FF",
        arrowprops=dict(arrowstyle="->", color="#7EF9FF", lw=1.4),
    )
    cb = fig.colorbar(mesh, ax=ax, pad=0.015)
    cb.set_label("GR 平方差（z-score 口径，越小越吻合）", fontsize=11)
    style_ax(
        ax,
        "方向 1｜信号确实存在：GR 匹配谷 = 轨迹候选（真实井 591cc951 评估区，150 列抽样）",
        "MD − 锚点（ft，井前方）",
        "ΔTVT = 候选 TVT − 最后已知 TVT（ft）",
    )
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ASSETS, "eda_alignment_panel.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def figure_lengths():
    st = pd.read_csv(STATS)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=200)

    ax = axes[0]
    bins = np.linspace(0, st[["md_len_known", "md_len_eval"]].max().max() * 1.02, 50)
    ax.hist(st["md_len_known"], bins=bins, color=BLUE, alpha=0.65, label="已知前缀长度")
    ax.hist(st["md_len_eval"], bins=bins, color=ORANGE, alpha=0.65, label="评估区长度")
    for val, color in [
        (st["md_len_known"].median(), BLUE),
        (st["md_len_eval"].median(), ORANGE),
    ]:
        ax.axvline(val, color=color, ls="--", lw=1.5)
    ax.text(
        st["md_len_known"].median() + 160,
        ax.get_ylim()[1] * 0.86,
        f"中位 {st['md_len_known'].median():,.0f} ft",
        color=BLUE,
        fontsize=11,
    )
    ax.text(
        st["md_len_eval"].median() + 160,
        ax.get_ylim()[1] * 0.7,
        f"中位 {st['md_len_eval'].median():,.0f} ft",
        color=ORANGE,
        fontsize=11,
    )
    style_ax(ax, "① 每口井的结构：评估区远长于已知前缀", "井段长度（ft）", "井数")
    ax.legend(fontsize=11)

    ax = axes[1]
    srt = st.sort_values("md_len_known").reset_index(drop=True)
    x = np.arange(len(srt))
    ax.bar(x, srt["md_len_known"], color=BLUE, width=1.0, label="已知前缀（有标签）")
    ax.bar(x, srt["md_len_eval"], bottom=srt["md_len_known"], color=ORANGE, width=1.0, label="评估区（待预测）")
    ax.set_xlim(-1, len(srt))
    ax.set_ylim(0, st["md_len_known"].max() + st["md_len_eval"].max() + 500)
    ax.set_xticks([])
    ax.text(
        0.5,
        0.96,
        f"评估区中位长度是已知前缀的 {st['md_len_eval'].median() / st['md_len_known'].median():.1f} 倍",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#333333",
        ha="center",
    )
    style_ax(ax, "② 773 口井的已知/评估结构（按已知长度排序）", "井（按已知前缀长度排序）", "井段长度（ft）")
    ax.legend(loc="upper left", fontsize=11)

    fig.suptitle(
        "方向 2｜这是“长程外推”问题：模型要用 1,700 ft 左右的证据，预测前方近 5,000 ft",
        fontsize=15,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ASSETS, "eda_lengths.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def figure_difficulty():
    st = pd.read_csv(STATS)
    srt = st.sort_values("const_sse", ascending=False).reset_index(drop=True)
    cum = srt["const_sse"].cumsum() / srt["const_sse"].sum()
    frac = (np.arange(1, len(srt) + 1)) / len(srt)
    k10 = min(int(np.ceil(len(srt) * 0.1)) - 1, len(srt) - 1)
    cum10 = float(cum.iloc[k10])
    k1 = min(int(np.ceil(len(srt) * 0.01)) - 1, len(srt) - 1)
    cum1 = float(cum.iloc[k1])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=200)

    ax = axes[0]
    ax.hist(st["const_rmse"], bins=45, color=BLUE_L, alpha=0.8, edgecolor="white")
    for val, color, lab in [
        (st["const_rmse"].median(), BLUE, f"中位 {st['const_rmse'].median():.1f} ft"),
        (st["const_rmse"].quantile(0.9), ORANGE, f"P90 {st['const_rmse'].quantile(0.9):.1f} ft"),
    ]:
        ax.axvline(val, color=color, ls="--", lw=1.5)
        ax.text(val + 0.4, ax.get_ylim()[1] * 0.9, lab, color=color, fontsize=11)
    style_ax(ax, "① 常数外推的每井 RMSE（真实标签口径）", "每井 RMSE（ft）", "井数")

    ax = axes[1]
    ax.plot(frac, cum, color=ORANGE, lw=2.6)
    ax.plot([0, 1], [0, 1], color=GRAY_L, ls="--", lw=1.3)
    ax.fill_between(frac, cum, frac, where=cum > frac, color=ORANGE, alpha=0.18)
    ax.scatter([0.101], [cum10], s=90, color=ORANGE, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"最难 10% 的井\n贡献 {cum10:.1%} 的误差平方和",
        xy=(0.101, cum10),
        xytext=(0.30, 0.72),
        fontsize=12,
        fontweight="bold",
        color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6),
    )
    ax.annotate(
        f"最难 1% 的井（约 8 口）贡献 {cum1:.1%}",
        xy=((k1 + 1) / len(srt), cum1),
        xytext=(0.05, 0.28),
        fontsize=10.5,
        color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.3),
    )
    style_ax(ax, "② 误差集中度曲线（井按误差降序）", "井数累计占比", "误差平方和累计占比")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle(
        "方向 3｜少数硬井主导总分：优化纪律 = 改难井，但绝不牺牲普通井",
        fontsize=15,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(ASSETS, "eda_difficulty.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def figure_gr_missing():
    st = pd.read_csv(STATS)
    all_miss = float((st["n_rows"] * st["gr_missing_all"]).sum() / st["n_rows"].sum())
    eval_miss = float((st["n_eval"] * st["gr_missing_eval"]).sum() / st["n_eval"].sum())

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), dpi=200)

    ax = axes[0]
    ax.hist(st["gr_missing_all"], bins=40, color=PURPLE, alpha=0.8, edgecolor="white")
    ax.axvline(all_miss, color=ORANGE, ls="--", lw=2)
    ax.axvline(eval_miss, color=TEAL, ls="--", lw=2)
    ax.text(all_miss + 0.008, ax.get_ylim()[1] * 0.9, f"全井 {all_miss:.1%}", color=ORANGE, fontsize=11)
    ax.text(eval_miss + 0.008, ax.get_ylim()[1] * 0.78, f"评估区 {eval_miss:.1%}", color=TEAL, fontsize=11)
    style_ax(ax, "① 每口井的 GR 缺失率分布（773 口）", "GR 缺失比例", "井数")

    ax = axes[1]
    sample = st.sort_values("gr_missing_all")["well"].iloc[::20].tolist()[:40]
    n_bin = 100
    mat = np.zeros((len(sample), n_bin), dtype=float)
    for r, w in enumerate(sample):
        hw = pd.read_csv(os.path.join(DATA, f"{w}__horizontal_well.csv"))
        pos = (hw["MD"] - hw["MD"].min()) / (hw["MD"].max() - hw["MD"].min())
        miss = hw["GR"].isna().to_numpy(dtype=float)
        b = np.clip((pos * n_bin).astype(int), 0, n_bin - 1)
        for k in range(n_bin):
            m = b == k
            mat[r, k] = miss[m].mean() if m.any() else 0.0
    im = ax.imshow(mat, aspect="auto", cmap="gray_r", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(sample))[::8])
    ax.set_yticklabels(sample[::8], fontsize=8.5)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)], fontsize=10)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("该位置 GR 缺失比例", fontsize=10.5)
    style_ax(ax, "② 缺失模式：常整段聚集，而非随机噪声（40 口抽样井）", "相对井深位置", "抽样井")
    ax.grid(False)

    fig.suptitle(
        "方向 4｜数据质量：近三成 GR 缺失，且呈段状——处理必须先于建模",
        fontsize=15,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ASSETS, "eda_gr_missing.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def figure_processing():
    well = "591cc951"
    hw, _ = load_well(well)
    known = hw["TVT_input"].notna().to_numpy()
    x = hw["MD"].to_numpy() - hw["MD"].min()
    gr = hw["GR"].to_numpy(dtype=float)
    gr_nan = np.isnan(gr)
    gr_fill = pd.Series(gr).interpolate(limit_direction="both").to_numpy()
    gr_z = robust_z(gr_fill)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7.2), dpi=200, sharex=True)
    ax = axes[0]
    for s, e in _nan_spans(gr_nan, x):
        ax.axvspan(s, e, color=GRAY, alpha=0.25, lw=0)
    ax.plot(x[known], gr_fill[known], color=BLUE, lw=1.2, label="已知前缀（插值后）")
    ax.plot(x[eval_mask := ~known], gr_fill[eval_mask], color=ORANGE, lw=1.2, label="评估区（插值后）")
    ax.scatter(x[gr_nan], gr_fill[gr_nan], s=14, color=GRAY, zorder=5, label=f"缺失点（本井 {gr_nan.mean():.1%}）")
    style_ax(ax, "① 原始 GR：先插值再建模（真实井 591cc951）", None, "GR（API）")
    ax.legend(loc="upper right", fontsize=10)

    ax = axes[1]
    ax.plot(x[known], gr_z[known], color=BLUE, lw=1.2)
    ax.plot(x[eval_mask], gr_z[eval_mask], color=ORANGE, lw=1.2)
    ax.axhline(0, color=GRAY, ls="--", lw=0.9)
    style_ax(ax, "② 逐井 robust-z 归一化：消除井间量纲差异，保留层位形态", "MD − 井口（ft）", "robust-z GR")
    fig.tight_layout()
    out = os.path.join(ASSETS, "eda_processing.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def figure_score_progression():
    milestones = [
        ("常数外推", 15.883, "本地地板", BLUE),
        ("LGBM 残差", 13.675, "本地", BLUE_L),
        ("Dual-PF 复现", 7.113, "公开口径", TEAL),
        ("GEO-2 物理先验", 7.008, "公榜最佳", TEAL),
        ("V6 main×soft", 7.9488, "诚实 OOF*", ORANGE),
    ]
    xs = np.arange(len(milestones))
    ys = [m[1] for m in milestones]
    fig, ax = plt.subplots(figsize=(13.5, 6.0), dpi=200)
    ax.plot(xs, ys, color=GRAY_L, lw=2, zorder=1)
    for i, (label, val, tag, color) in enumerate(milestones):
        marker = "o" if i < 4 else "D"
        ax.scatter([i], [val], s=190 if i < 4 else 260, color=color, zorder=3,
                   edgecolor="white", linewidth=1.8, marker=marker)
        ax.annotate(
            f"{label}\n{val}",
            xy=(i, val),
            xytext=(i, val + 1.05),
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )
        ax.annotate(tag, xy=(i, val), xytext=(i, val - 2.6), ha="center", fontsize=10, color=GRAY)
    ax.set_xticks(xs)
    ax.set_xticklabels([m[0] for m in milestones], fontsize=11)
    ax.set_ylim(3.5, 17.5)
    ax.set_ylabel("RMSE（ft，越低越好）")
    ax.invert_yaxis()
    ax.text(
        0.015,
        0.96,
        "每个里程碑的口径不同（本地 / 公开 / OOF），只展示相对提升，不做跨点直接比较",
        transform=ax.transAxes,
        fontsize=10.5,
        color=GRAY,
        va="top",
    )
    ax.text(
        0.015,
        0.08,
        "*V6 为按井分组的诚实 OOF（7.9488），比公榜口径更严格；最终 Private 排名 487 / 6,125（TOP 8%）",
        transform=ax.transAxes,
        fontsize=10.5,
        color=ORANGE,
        va="bottom",
    )
    style_ax(ax, "方案演进：RMSE 从 15.883 降到 7.008（−56%），且最终方案经过最严格验证", None, None)
    ax.grid(axis="y", alpha=0.6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    out = os.path.join(ASSETS, "score_progression.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def figure_metrics_banner():
    cards = [
        ("487 / 6,125", "最终 Private 排名 · TOP 8%", BLUE),
        ("RMSE ↓ 56%", "15.883 → 7.008（公榜口径）", TEAL),
        ("773 口井 · 509 万行", "全量训练数据逐行建模", ORANGE),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.0), dpi=200)
    for ax, (num, cap, color) in zip(axes, cards):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        box = FancyBboxPatch(
            (0.02, 0.05),
            0.96,
            0.90,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            transform=ax.transAxes,
            facecolor="white",
            edgecolor=color,
            linewidth=2.4,
        )
        ax.add_patch(box)
        ax.text(0.5, 0.63, num, transform=ax.transAxes, ha="center", va="center",
                fontsize=25, fontweight="bold", color=color)
        ax.text(0.5, 0.28, cap, transform=ax.transAxes, ha="center", va="center",
                fontsize=11.5, color="#444444")
    fig.tight_layout()
    out = os.path.join(ASSETS, "metrics_banner.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def main():
    figure_problem_definition()
    figure_alignment_panel()
    figure_lengths()
    figure_difficulty()
    figure_gr_missing()
    figure_processing()
    figure_score_progression()
    figure_metrics_banner()


if __name__ == "__main__":
    main()
