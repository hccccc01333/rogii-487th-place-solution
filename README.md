# 第 489 名解法：main×soft 融合 + Full773 V6 提交内核 — ROGII 井筒地质预测

**队伍：Rainfun** · 最终 Private 排名：**489 / 1,000+**

Kaggle 竞赛：[ROGII - Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)

---

## 总结

我们的方案把问题拆成三层：**物理/参数化候选负责“提出合理的轨迹”**，
**软门控融合负责“决定每一处该信谁”**，**提交内核负责“稳定、可复现地写出答案”**。

最终方案 V6（main × soft merge，本地版本号 `new_0604`）的本地诚实 OOF 为
**7.9488**，是我们在全量 773 口井按井分组外折口径下的最优单模型；配合
Full773 参数化提交内核（P3/P4 家族四个候选），最终提交获得 Private 第 489 名。

更重要的收获来自工程侧：这场比赛只接受 Notebook 提交，而提交文件的**写出方式决定生死**
——v1–v4 全部因为“原始字节写文件”被判格式错误，V6 改为
`pandas to_csv` + 与 `sample_submission` 合并后才真正出分。本仓库保留的就是这套
最终固化的 **V6** 版本。

![方案演进](assets/score_progression.png)

---

## 1. 问题与验证

### 1.1 任务

- 每口水平井给出 MD / XYZ / GR 测井、已知前缀段 TVT（`TVT_input`）以及一口配对 typewell；
- 目标：预测评估区（`TVT_input` 为空的后缀段）逐行 TVT；
- 指标：评估区全部行的 pooled RMSE（少数长而难的井会主导总分）。

### 1.2 数据规模

- 训练井：773 口；
- 隐藏测试：约 200 口；公开样例只有 3 口井（14151 行）——**样例规模远小于真实隐藏集**，
  所有推理与运行时设计都按真实规模假设执行。

### 1.3 验证协议（诚实 OOF）

- 按井 GroupKFold 5 折，同一口井的所有行、所有伪前缀样本同折；
- 候选特征遵守外折合同：候选对其生产者是 OOF 的，下游训练不再触碰验证折；
- 所有报告同时给出：逐折结果、pooled、最差 10% 井、相对基线胜率；
- 公榜只当作固定偏移参考，不用于调参。

---

## 2. 最终方案：V6 详解

![V6 流水线](assets/pipeline.png)

### 2.1 main 分支：参数化主模型

我们维护了一个“地质/轨迹参数化”主模型家族，全部按折外口径训练：

| 候选 | 组成 | 诚实 OOF |
|------|------|--------:|
| P3 | 参数化线性轨迹 | 7.4341 |
| P4 | 参数化二次轨迹 | 7.2794 |
| P4+Ridge | 参数化 + Ridge 融合（best OOF composite） | **7.2164** |
| P3+P4 | 保守曲率对冲 | 7.2924 |

### 2.2 soft 分支：晚发/易稳软门控

在 main 之外，我们训练了井级分类特征 `pred_late`（晚发井）与 `pred_easy`（易稳井），
全部来自前缀/轨迹/typewell 的合法可见特征，输出为折外预测。

### 2.3 融合：0.6 × main + 0.4 × soft

```text
pred_v6 = 0.6 × main + 0.4 × soft
```

固定权重、不按井硬切换，避免在小样本上过拟合；本地诚实 OOF = **7.9488**。
后续实验证明：更复杂的“灾难井专用选择器”和“第三证据源”在 oracle 上很强，
但收益回归器无法在合法特征下稳定区分灾难井，全局加入反而更差——因此最终保持纯软融合。

### 2.4 Full773 提交内核（V6）

真正出分的提交由四个资产驱动的内核完成：

```text
PARAM_*_FULL_lf.csv（Full773 候选）
   → SHA-256 校验
   → pandas 读入
   → 与 sample_submission 合并
   → pandas to_csv 写出 submission.csv
```

- 候选来自 `full773-submission-candidates-lf` 私有 dataset（`PARAM_*_FULL_lf.csv`）；
- 内核在本地 3 口井冒烟口径下验证 14151 行、id 顺序一致、无 NaN；
- 真实隐藏测试由 Kaggle 侧替换完整候选后直接运行，推理代码不变。

---

## 3. 分数演进

| 阶段 | 方案 | 分数 | 口径 |
|------|------|-----:|------|
| 地板 | const 基线 | 15.883 | Public |
| 第一版 ML | gated LGBM | 13.675 | Public |
| 公开管线对齐 | Dual-PF | 7.113 | Public |
| 构造面先验 | GEO-2 | 7.008 | Public（冻结基线） |
| **最终方案** | **V6（main × soft merge）** | **7.9488** | 本地诚实 OOF |
| 最终成绩 | V6 Full773 提交内核 | — | Private 第 489 名 |

![方案演进图](assets/score_progression.png)

---

## 4. 关键失败与修复

### 4.1 提交写入方式（最痛的教训）

v1–v4 的 Full773 内核全部 `COMPLETE` 但 `total_bytes=0`，被判定为“提交格式错误”：

- 错误：把候选 CSV 以“原始字节拷贝”写文件；
- 修复：V6 改用 `pandas to_csv`，并先与 `sample_submission` 合并、断言 id 唯一/无 NaN/有限；
- 结论：**在 Notebook-only 比赛中，写出路径与模型同样重要**。

### 4.2 隐藏集规模

公开样例 14151 行（3 井）远小于真实隐藏集。任何“预先算好结果”的方案都会被真实规模直接
打死；所有推理设计都按“每井耗时 × 井数 ≤ 时限 × 0.75”的墙生存预算执行。

### 4.3 OOF 纪律

反复验证：本地 OOF 有增益并不等于公榜会兑现。原因包括候选的嵌套 OOF 泄漏、
best-of-many 偏差、以及小样本公榜噪声。因此我们坚持外折合同 + null 对照 +
失败分类（死胡同/角色错误/时机错误/成本放弃），只有三指标同向的改进才被保留。

---

## 5. 如何用 Agent 完成这场比赛

这场比赛不是“人写代码、AI 帮忙查资料”，而是**把 AI 当作一支可治理的研发团队**来使用：
Codex 负责执行与实验，网页版 GPT-5.6 负责导演与决策，多个子 agent 负责并行探索与评审。

![Agent 协作工作流](assets/agent_workflow.png)

### 5.1 协作日志制度（人机对话全过程留痕）

从第一天开始，我们维护了一份 `AI协作日志`，**每一次交互都记录三件事**：

1. 用户提示词（原样）；
2. AI 应用思路（为什么这样做、边界在哪里）；
3. AI 回答摘要与产出物（文件、指标、提交 ref）。

这份日志贯穿全程：从“指向赛题目录”开始，到“多子 agent 方案讨论”、“评价 agent 只读评审”、
“逐次提交记录”，再到最终的四路线构建。它让每一次 AI 决策都可回放、可审计、可复盘。

### 5.2 三线并行的 agent 架构

- **Track A — 诚实建模**：Codex 构建并维护“全量 773 井、按井外折、不泄漏”的模型注册表与
  表现矩阵；
- **Track B — 公共代码血统**：对公开高分 notebook 做模块级差异审计与可迁移机制分析；
- **Track I — 公共情报**：爬取并结构化讨论区、解法、代码链接，形成情报库。

三个 Track 并行推进，互相独立，避免单一路线的确认偏误。

### 5.3 子 agent 与评审 agent

- **多子 agent 方案讨论**：在规则约束下并行讨论多条技术路线，汇总后再由人拍板；
- **只读评审 agent**：对当前工作产出结构化评价（分数、证据、优先级），
  明确“哪些方向不该继续投入”，避免 AI 自我确认；
- **GPT-5.6 导演协议**：每个冲刺周期生成“给导演的提示词 + 证据包”，
  由导演评审后给出下一轮执行指令，形成 人 → Codex → 导演 → 人 的闭环。

### 5.4 纪律化实验管理

- 实验板：一次改动一张可比表，指标口径锁定；
- NO-GO 注册表：失败的实验必须记录 `result / evidence / reason / reopen_condition`，
  只冻结上下文、不永久封杀组件；
- 确定性审计：Notebook 连续两次运行 SHA 一致，提交前校验输出行数与顺序；
- 提交预算：每日 ≤5 次，同一版本只交一枪，ERROR 即停。

### 5.5 这段经历说明了什么

AI 的价值不只是“生成代码”，而是可以承载一整套**工程纪律**：日志、门禁、审计、
决策留痕、失败分类、重开条件。只要给 agent 定义好验证合同与审批点，它就能持续、
稳定地把“实验 → 结论 → 提交”的循环跑下去——这正是我们在本场比赛中证明的
agent 使用能力。

---

## 6. 目录结构

```text
.
├── README.md
├── assets/                    # README 可视化（分数演进 / 流水线 / Agent 工作流）
├── 概述.md / 数据.md / 规则.md  # 竞赛背景资料
├── notebooks/
│   ├── full773_v6/            # Full773 v6 提交内核（4 个候选 + v6_meta）
│   │   ├── rogii-full773-p3-linear-v1/         # notebook + submission.csv
│   │   ├── rogii-full773-p4-quadratic-v1/
│   │   ├── rogii-full773-p4-ridge-blend-v1/
│   │   ├── rogii-full773-p3-p4-blend-v1/
│   │   └── rogii-full773-p3-linear-v1.ipynb    # v6 元数据版
│   └── main_soft_v6/
│       └── rogii-main-soft-full-v6.ipynb       # main×soft merge V6 全量 notebook
└── scripts/
    ├── blend_v6_merge_learned.py   # V6 融合
    ├── build_v6_full_notebook.py   # V6 全量 notebook 构建
    └── make_readme_assets.py       # README 配图生成脚本
```

## 7. 复现说明

提交内核是资产驱动的：真实隐藏测试需要替换候选 CSV
（原私有 dataset：`afterrain42/full773-submission-candidates-lf`，
即 `PARAM_*_FULL_lf.csv` 的完整版）后直接运行 notebook。
本地 `submission.csv` 为 3 口样例井冒烟输出，用于验证行数 / id 顺序 / 无 NaN。
所有指标均注明口径（本地 OOF / Public / Private），不含任何私有数据。
