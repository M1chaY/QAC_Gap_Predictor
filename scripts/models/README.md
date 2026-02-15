# models - 模型训练脚本

本目录包含 GNN 模型的超参数搜索和训练脚本。

## 工作流程

### 1. 超参数搜索

使用 Optuna 框架自动搜索最佳网络结构：

```bash
python scripts/models/optuna_gnn_framework.py
```

### 2. 最终训练

读取搜索结果进行完整训练：

```bash
python scripts/models/gnn_pretrain_onqm9.py
```

## 文件说明

| 文件 | 功能 |
| ---- | ---- |
| `optuna_gnn_framework.py` | Optuna 超参数搜索框架 |
| `gnn_pretrain_onqm9.py` | QM9 数据集预训练脚本 |
| `finetune_gnn.py` | QAC 数据集微调脚本 |
| `qac_gap_predictor.py` | Gap 预测应用脚本 |
| `pretrain_gnn.py` | (已废弃) 旧版预训练脚本 |

## 超参数搜索 (optuna_gnn_framework.py)

### 搜索空间

| 参数 | 范围 | 说明 |
| ---- | ---- | ---- |
| `hidden_channels` | [32, 64, 128, 256] | 隐藏层维度 |
| `num_heads` | [2, 4, 8] | GAT 注意力头数 |
| `num_gat_layers` | [1, 4] | GAT 层数 |
| `num_mlp_layers` | [2, 5] | MLP 层数 |
| `dropout` | [0.1, 0.5] | Dropout 概率 |
| `lr` | [1e-4, 1e-2] | 学习率 (对数尺度) |
| `weight_decay` | [1e-6, 1e-3] | 权重衰减 (对数尺度) |

### 搜索策略

- 数据集划分：70% 训练 / 15% 验证 / 15% 测试
- 优化目标：验证集 MAE
- 每个 trial：最多 300 次迭代
- 早停耐心值：50
- 采样器：TPE (Tree-structured Parzen Estimator)
- 剪枝器：Median Pruner

### 搜索输出文件

| 文件 | 说明 |
| ---- | ---- |
| `models/optuna_best_config.json` | 最佳超参数配置 (JSON) |
| `models/optuna_search_report.md` | 搜索报告 (Markdown) |

## 预训练 (gnn_pretrain_onqm9.py)

### 训练策略

- 读取 `optuna_best_config.json` 中的网络结构
- 数据集划分与搜索阶段一致（相同随机种子）
- 最大迭代次数：1000
- 早停耐心值：100
- 学习率调度：ReduceLROnPlateau

### 预训练输出文件

| 文件 | 说明 |
| ---- | ---- |
| `models/qm9_pretrained.pt` | 预训练模型权重 |
| `models/qm9_pretrain_loss_curve.png` | 损失曲线图 |
| `models/qm9_pretrain_scatter.png` | 预测散点图 |

## 依赖关系

```text
input_graph_preparation.py
        ↓
optuna_gnn_framework.py
        ↓
gnn_pretrain_onqm9.py
        ↓
finetune_gnn.py
```

## 注意事项

1. 运行前请确保已生成 `data/qm9_prepared.joblib` 数据集
2. Optuna 搜索可能需要较长时间，建议使用 GPU
3. 所有脚本使用固定随机种子 (42) 以确保可复现性
