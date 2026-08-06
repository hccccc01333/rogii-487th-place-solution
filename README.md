# ROGII Wellbore Geology Prediction — 489th Place Solution (V6)

Kaggle 竞赛：[ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

最终排名：**489 / 1,000+**（Private LB） · 队伍：hongzhouzhou（afterrain42）

## 赛题简述

- 每口水平井给出 MD / XYZ / GR 测井、已知前缀段 TVT（`TVT_input`）以及一口配对 typewell；
- 目标：预测评估区（`TVT_input` 为空的后缀段）逐行 TVT；
- 指标：评估区全部行的 pooled RMSE；
- 训练井 773 口，隐藏测试约 200 口（公开样例只有 3 口井，14151 行）。

## 来龙去脉（方案演进）

这场比赛我们走了一条“先地板、后物理、再固化”的路线：

| 阶段 | 提交 / 方案 | Public | 说明 |
|------|-------------|-------:|------|
| 地板 | const 基线 | 15.883 | 最后已知 TVT 常数外推 |
| 第一版 ML | gated LGBM | 13.675 | 残差模型 + hard_score 门控 |
| 公开管线复现 | Dual-PF 对齐 | 7.113 | 复现公开 7.061 家族并本地对齐 |
| 构造面先验 | GEO-2 | 7.008 | 多地层平面 + 前缀校准（冻结基线） |
| 公开最佳 | tip corridor 系列 | **6.440** | PF 走廊 + 分支选择的公开最佳 |
| **最终方案** | **V6（main × soft merge）** | — | 本地诚实 OOF **7.9488**，最终 Private 第 489 名 |

关键教训之一：本比赛只接受 Notebook 提交，且**提交文件的写出方式决定生死**。
v1–v4 的 Full773 内核全部因“原始字节写文件”被判格式错误；V5/V6 改为
`pandas to_csv` + 与 `sample_submission` 合并后，提交才真正出分。
本仓库保留的是最终固化的 **V6** 版本。

## V6 方案结构

V6 = **main × soft 门控融合**（`new_0604`），核心是“让强主模型当底盘，
让软门控按井的晚发/易稳特征做小幅修正”：

- **main 分支**：地质/轨迹参数化主模型（P3 linear / P4 quadratic / P4+Ridge blend / P3+P4 blend），
  逐折 GroupKFold 训练；
- **soft 分支**：`pred_late / pred_easy` 折外分类特征，按 `0.6×main + 0.4×soft` 固定权重软融合；
- **Full773 提交候选**：四个参数化候选（`PARAM_P3_LINEAR` / `PARAM_P4_QUADRATIC` /
  `PARAM_P4_RIDGE_BLEND` / `PARAM_P3_P4_BLEND`），诚实 OOF 分别为
  7.434 / 7.279 / 7.216 / 7.292；
- **提交内核（v6）**：读私有 dataset 候选 → SHA-256 校验 → `pandas to_csv` 写出，
  与 `sample_submission` 合并，最终 `submission.csv` 14151 行（本地 3 井冒烟口径）。

所有 OOF 均按井 GroupKFold、外折合同计算，不混入目标井后缀真值。

## 目录

```text
.
├── README.md
├── 概述.md / 数据.md / 规则.md   # 竞赛背景资料
├── notebooks/
│   ├── full773_v6/              # Full773 v6 提交内核（4 个候选 + v6_meta）
│   │   ├── rogii-full773-p3-linear-v1/         # notebook + submission.csv
│   │   ├── rogii-full773-p4-quadratic-v1/
│   │   ├── rogii-full773-p4-ridge-blend-v1/
│   │   ├── rogii-full773-p3-p4-blend-v1/
│   │   └── rogii-full773-p3-linear-v1.ipynb    # v6 元数据版
│   └── main_soft_v6/
│       └── rogii-main-soft-full-v6.ipynb       # main×soft merge V6 全量 notebook
└── scripts/
    ├── blend_v6_merge_learned.py   # V6 融合
    └── build_v6_full_notebook.py   # V6 全量 notebook 构建
```

## 复现说明

提交内核是资产驱动的：真实隐藏测试需要替换候选 CSV
（原私有 dataset：`afterrain42/full773-submission-candidates-lf`，
即 `PARAM_*_FULL_lf.csv` 的完整版）后直接运行 notebook。
本地 `submission.csv` 为 3 口样例井冒烟输出，用于验证行数 / id 顺序 / 无 NaN。

## 诚实性声明

- 本仓库**不包含**：复盘文档、通用 Kaggle 工作流、知识库、任何私有数据；
- 所有指标均注明口径（本地 OOF / Public / Private）；
- 公开最佳 6.440 与最终 Private 排名 489 分别来自不同提交代际，按实际榜单记录。
