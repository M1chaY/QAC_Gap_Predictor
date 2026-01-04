# model - 模型定义模块

本目录包含 GNN 模型定义、训练工具、评估和可视化功能。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `gap_gnn.py` | HOMO-LUMO Gap 预测 GNN 模型定义 | <150 |
| `training.py` | 训练和评估工具函数 | <150 |
| `evaluation.py` | 回归模型评估指标计算 | <150 |
| `visualization.py` | 训练损失曲线可视化 | <150 |
| `scatter_plot.py` | 实际值vs预测值散点图 | <150 |

## 模型架构

`GapPredictionGNN` 是一个轻量级图注意力网络，适用于小数据集：

- **图卷积**: 2 层 GAT (Graph Attention Network)
- **池化**: Mean + Max 全局池化
- **回归头**: 3 层 MLP
- **正则化**: Dropout + BatchNorm

## 使用示例

### 定义模型

```python
import torch
from src.model import GapPredictionGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GapPredictionGNN(
    num_node_features=9,
    hidden_channels=80,
    num_global_features=3,
    num_heads=4
).to(device)
```

### 训练和评估

```python
from src.model import train_epoch, evaluate, compute_loss

loss = train_epoch(model, train_loader, optimizer, device)
r2, mae, rmse, preds, targets = evaluate(model, test_loader, device)
```

### 评估指标

```python
from src.model import calculate_metrics, metrics_to_dataframe

r2, mae, mape, rmse = calculate_metrics(y_true, y_pred)
df = metrics_to_dataframe(y_train, y_train_pred, y_test, y_test_pred, "GNN")
```

### 可视化

```python
from src.model import plot_loss_curves, plot_actual_vs_predicted

plot_loss_curves(train_losses, test_losses, "loss.png")
plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, "scatter.png")
```
