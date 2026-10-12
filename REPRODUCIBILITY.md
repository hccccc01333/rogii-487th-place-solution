# Reproducibility / 复现说明

这个仓库侧重**公开方法与部分提交代码**，不是完整的 Full773 训练数据、特征仓库和资产快照。不能承诺从零一键再现 Private LB 排名。

## 现有内容

- `README.md`：问题定义、特征、融合、验证原则及实验总结。
- `notebooks/main_soft_v6/`：主线与软门控的提交 notebook。
- `notebooks/full773_v6/`：候选提交 notebook 和 `kernel-metadata.json`。
- `scripts/`：融合、Notebook 构建及图表脚本。

## 外部数据和依赖

1. 从 [ROGII Kaggle 竞赛页面](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) 获取你有权使用的数据，并遵守比赛条款。
2. `notebooks/full773_v6/kernel-metadata.json` 引用了 Kaggle 数据集 `afterrain42/full773-submission-candidates-lf`（元数据标为 private），仓库**未提供公开下载保证**。
3. 完整提交还依赖训练所得候选 CSV、特征和模型资产。这些文件并未全部收录，不能直接复制排名结果。
4. 部分 `scripts/` 文件引用缺失的本地 `project/src/`、`project/artifacts/` 或 Windows 路径；它们是研究快照，不是开箱即用工具。
5. 本仓库尚无锁定的 requirements / conda 环境文件；Notebook 依赖需按具体单元检查（如 numpy、pandas、scipy、scikit-learn、lightgbm）。

## 现在可以核验的内容

无需下载竞赛数据，可以在仓库根目录检查 Python 源码是否可编译：

```bash
python -m compileall -q scripts
```

这只是语法检查，不验证运行时依赖、数据处理、预测结果或排名。若你已合法获取所需数据/资产，可以根据对应 Notebook 的输入路径在 Kaggle 环境逐步执行并记录版本、输入文件哈希和评价口径。

## 评估约定

- 区分 Public LB、Private LB 和按井 GroupKFold 的 OOF；不同口径的分数不能直接比较。
- 记录数据版本、特征来源、fold 划分、候选资产的 SHA-256 与随机种子。
- 若无法复现，请通过 Issue 写明卡在哪一步；补充复现说明本身也是贡献。

## 授权说明

MIT 只覆盖维护者有权许可的原创软件和文档；竞赛数据、来源于第三方的 Notebook/库、品牌素材和外部训练资产不随本仓库自动获得 MIT 授权。
