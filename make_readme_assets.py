# -*- coding: utf-8 -*-
"""Generate README assets: score progression, pipeline, agent workflow."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)


def box(ax, x, y, w, h, text, fc="#171c26", ec="#20beff", tc="white", fs=10):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.5, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#20beff"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=16, linewidth=1.8, color=color)
    ax.add_patch(a)


# ---------- 1. Score progression ----------
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
stages = ["const\n基线", "gated\nLGBM", "Dual-PF\n公开管线", "GEO-2\n构造面先验", "V6\nmain×soft"]
scores = [15.883, 13.675, 7.113, 7.008, 7.9488]
colors = ["#8b95a7", "#8b95a7", "#8b95a7", "#8b95a7", "#20beff"]
x = range(len(stages))
ax.bar(x, scores, color=colors, width=0.55, zorder=3)
for i, s in enumerate(scores):
    note = "（本地诚实 OOF）" if i == 4 else "（Public）"
    ax.text(i, s + 0.35, f"{s:.3f}\n{note}", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel("RMSE（越低越好）", fontsize=11)
ax.set_ylim(0, 18)
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.set_title("ROGII 方案演进：从 15.9 到最终 V6（最终 Private 487 / 6,125，TOP 8%）", fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "score_progression.png", bbox_inches="tight")
plt.close(fig)

# ---------- 2. Pipeline ----------
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.2)
ax.axis("off")
box(ax, 0.3, 3.6, 2.1, 1.1, "main 分支\nP3 linear / P4 quadratic\nP4+Ridge blend / P3+P4 blend\nOOF 7.434 / 7.279 / 7.216 / 7.292", fc="#10151f")
box(ax, 0.3, 1.6, 2.1, 1.1, "soft 分支\npred_late / pred_easy\n折外分类特征", fc="#10151f")
box(ax, 3.2, 2.9, 2.0, 1.2, "main × soft 融合\n0.6 × main + 0.4 × soft\n本地 OOF 7.9488", fc="#14212e")
arrow(ax, 2.4, 4.1, 3.2, 3.6)
arrow(ax, 2.4, 2.1, 3.2, 3.2)
box(ax, 5.7, 2.9, 1.9, 1.2, "Full773 候选\nPARAM_*_FULL_lf.csv\n+ SHA-256 校验", fc="#14212e")
arrow(ax, 5.2, 3.5, 5.7, 3.5)
box(ax, 8.0, 2.9, 1.8, 1.2, "V6 提交内核\npandas to_csv\n+ sample merge", fc="#20beff")
arrow(ax, 7.6, 3.5, 8.0, 3.5)
box(ax, 3.2, 0.6, 3.2, 1.0, "按井 GroupKFold 5 折\n诚实外折 OOF，不混入后缀真值", fc="#10151f", ec="#34d399")
arrow(ax, 4.8, 1.6, 4.8, 2.9)
ax.set_title("V6 方案流水线：参数化主模型 × 软门控 → Full773 候选 → 提交内核", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "pipeline.png", bbox_inches="tight")
plt.close(fig)

# ---------- 3. Agent workflow ----------
fig, ax = plt.subplots(figsize=(10, 5.4), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.4)
ax.axis("off")
box(ax, 0.3, 4.2, 2.0, 0.9, "用户需求\n（比赛/提交/复盘）", fc="#10151f")
box(ax, 2.9, 4.2, 2.4, 0.9, "Codex 执行\n（实验/门禁/日志/提交）", fc="#14212e")
box(ax, 5.8, 4.2, 2.2, 0.9, "网页 GPT-5.6 导演\n（评审/决策/交接）", fc="#14212e")
box(ax, 8.4, 4.2, 1.4, 0.9, "提交/复盘", fc="#20beff")
arrow(ax, 2.3, 4.65, 2.9, 4.65)
arrow(ax, 5.3, 4.65, 5.8, 4.65)
arrow(ax, 8.0, 4.65, 8.4, 4.65)
box(ax, 0.3, 2.7, 2.0, 0.9, "AI 协作日志\n提示词/思路/回答\n逐条记录", fc="#10151f", ec="#34d399")
box(ax, 2.9, 2.7, 2.4, 0.9, "实验板 + 门禁\nGroupKFold / OOF\nnull 对照", fc="#10151f", ec="#34d399")
box(ax, 5.8, 2.7, 2.2, 0.9, "NO-GO 注册表\n失败分类 + 重开条件", fc="#10151f", ec="#34d399")
box(ax, 8.4, 2.7, 1.4, 0.9, "确定性审计\nSHA 一致", fc="#10151f", ec="#34d399")
arrow(ax, 1.3, 4.2, 1.3, 3.6)
arrow(ax, 4.1, 4.2, 4.1, 3.6)
arrow(ax, 6.9, 4.2, 6.9, 3.6)
arrow(ax, 9.1, 4.2, 9.1, 3.6)
box(ax, 2.4, 1.0, 5.2, 1.0, "三线并行：Track A 诚实建模 / Track B 公共代码血统 / Track I 公共情报\n子 agent 方案讨论 + 只读评审 agent", fc="#10151f", ec="#f59e0b")
ax.set_title("Agent 协作工作流：可治理的 AI 研发流水线", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "agent_workflow.png", bbox_inches="tight")
plt.close(fig)

# ---------- 4. Key metrics banner ----------
fig, ax = plt.subplots(figsize=(10, 2.6), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 2.6)
ax.axis("off")
metrics = [
    ("487 / 6,125", "最终 Private 排名\nTOP 8%", "#20beff"),
    ("-56%", "预测误差降幅\n15.883 → 7.008", "#34d399"),
    ("773", "训练井全量外折\n不泄漏验证", "#f59e0b"),
    ("7.9488", "V6 本地诚实 OOF\n按井 5 折", "#a78bfa"),
]
x = 0.35
for big, small, color in metrics:
    box(ax, x, 0.35, 2.15, 1.9, "", fc="#10151f", ec=color)
    ax.text(x + 1.075, 1.55, big, ha="center", va="center", color=color,
            fontsize=22, fontweight="bold")
    ax.text(x + 1.075, 0.85, small, ha="center", va="center", color="#c7d0dd",
            fontsize=10)
    x += 2.35
ax.set_title("项目关键指标", fontsize=13, pad=10)
fig.tight_layout()
fig.savefig(OUT / "metrics_banner.png", bbox_inches="tight")
plt.close(fig)

# ---------- 5. Problem schematic ----------
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.2)
ax.axis("off")
# trajectory
import numpy as np
md = np.linspace(0.5, 9.5, 200)
tvt = 2.6 + 0.55 * np.sin(md / 2.2) - 0.12 * md
ax.plot(md, tvt, color="#20beff", lw=3, zorder=3)
# prefix zone
ax.axvspan(0.5, 4.2, color="#34d399", alpha=0.12, zorder=1)
# eval zone
ax.axvspan(4.2, 9.5, color="#f87171", alpha=0.10, zorder=1)
ax.axvline(4.2, color="#f87171", lw=2, ls="--", zorder=2)
ax.text(2.35, 3.75, "已知段（前缀）\nTVT 已知，可标定", ha="center", fontsize=11, color="#34d399")
ax.text(6.85, 3.75, "评估区（待预测）\n目标：预测逐行 TVT", ha="center", fontsize=11, color="#f87171")
ax.text(4.35, 1.1, "PS（预测起点）", ha="left", fontsize=10, color="#f87171")
ax.text(5.3, 2.2, "水平井轨迹", fontsize=10, color="#20beff")
ax.set_title("问题示意：水平井轨迹 + 已知前缀 → 预测评估区", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "problem_schematic.png", bbox_inches="tight")
plt.close(fig)

# ---------- 6. Difficulty concentration (factual stacked bar) ----------
fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
ax.barh([0], [42.2], color="#8b95a7", height=0.5, label="其余 90% 的井（42.2%）")
ax.barh([0], [57.8], left=[42.2], color="#f87171", height=0.5,
        label="最难 10% 的井（57.8%）")
for xpos, txt in ((21.1, "42.2%"), (71.1, "57.8%")):
    ax.text(xpos, 0, txt, ha="center", va="center", color="white", fontsize=13,
            fontweight="bold")
ax.set_yticks([])
ax.set_xlim(0, 100)
ax.set_xlabel("总误差平方和（SSE）占比", fontsize=11)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=10)
ax.set_title("难度集中度：最难 10% 的井贡献了 57.8% 的总误差", fontsize=13)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "eda_difficulty.png", bbox_inches="tight")
plt.close(fig)

# ---------- 7. 5-fold GroupKFold protocol ----------
fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.6)
ax.axis("off")
for i in range(5):
    x0 = 0.5 + i * 1.85
    for j in range(5):
        color = "#f87171" if j == i else "#20beff"
        box(ax, x0, 0.5 + j * 0.52, 1.35, 0.36, "", fc="#10151f", ec=color)
    ax.text(x0 + 0.675, 3.2, f"折{i+1}", ha="center", fontsize=11, color="white")
ax.text(5.0, 0.12, "每折：蓝色 = 训练井组，红色 = 验证井组；同一口井的所有行永远在同一折",
        ha="center", fontsize=10, color="#c7d0dd")
ax.set_title("验证协议：按井分组的 5 折交叉验证", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "cv_protocol.png", bbox_inches="tight")
plt.close(fig)

# ---------- 8. End-to-end workflow overview ----------
fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.2)
ax.axis("off")
steps = [
    ("1 数据", "773 口井\n测井+轨迹+参考井"),
    ("2 分析", "EDA\n难度结构/信号"),
    ("3 特征", "清洗对齐\n三类特征"),
    ("4 建模", "候选+融合\nV6"),
    ("5 验证", "5折外折\n防泄漏门禁"),
    ("6 提交", "内核+审计\n复盘反思"),
]
x = 0.25
for title, desc in steps:
    box(ax, x, 0.7, 1.5, 1.8, f"{title}\n\n{desc}", fc="#14212e", ec="#20beff", fs=9)
    if x < 8.5:
        arrow(ax, x + 1.55, 1.6, x + 1.85, 1.6)
    x += 1.65
ax.set_title("端到端分析流程：从原始数据到可复现提交", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "workflow_overview.png", bbox_inches="tight")
plt.close(fig)

print("ASSETS_OK", sorted(p.name for p in OUT.glob("*.png")))
