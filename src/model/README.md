# model - 模型定义模块

本目录包含 GNN 模型定义、训练工具、评估和可视化功能。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `gap_gnn.py` | HOMO-LUMO Gap 预测 GNN 模型定义（支持可配置层数） | <150 |
| `training.py` | 训练和评估工具函数 | <150 |
| `evaluation.py` | 回归模型评估指标计算 | <150 |
| `loss_curves.py` | 训练损失曲线可视化 | <150 |
| `scatter_plot.py` | 实际值vs预测值散点图 | <150 |

## 模型架构

`GapPredictionGNN` 是一个可配置的图注意力网络，支持 Optuna 超参数搜索：

- **图卷积**: 可配置层数的 GAT (Graph Attention Network)
- **池化**: Mean + Max 全局池化
- **回归头**: 可配置层数的 MLP
- **正则化**: Dropout + BatchNorm

### 可配置参数

| 参数 | 说明 | 默认值 |
| ---- | ---- | ------ |
| `num_node_features` | 节点特征维度 | - |
| `hidden_channels` | 隐藏层维度 | 64 |
| `num_global_features` | 全局分子描述符数量 | 3 |
| `num_heads` | GAT注意力头数 | 4 |
| `num_gat_layers` | GAT层数 | 2 |
| `num_mlp_layers` | MLP层数 | 3 |
| `dropout` | Dropout概率 | 0.2 |

## 使用示例

### 定义模型

```python
import torch
from src.model import GapPredictionGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用默认参数
model = GapPredictionGNN(
    num_node_features=9,
    hidden_channels=64,
    num_global_features=3,
    num_heads=4
).to(device)

# 使用自定义层数
model = GapPredictionGNN(
    num_node_features=9,
    hidden_channels=128,
    num_global_features=3,
    num_heads=4,
    num_gat_layers=3,
    num_mlp_layers=4,
    dropout=0.3
).to(device)
```

### 从配置文件创建模型

```python
config = {
    'hidden_channels': 128,
    'num_heads': 4,
    'num_gat_layers': 3,
    'num_mlp_layers': 4,
    'dropout': 0.3
}

model = GapPredictionGNN.from_config(
    config, 
    num_node_features=9, 
    num_global_features=3
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

# 绘制损失曲线
plot_loss_curves(train_losses, test_losses, "loss.png")

# 绘制预测散点图（支持训练集和测试集对比）
plot_actual_vs_predicted(
    y_train, y_train_pred, y_test, y_test_pred,
    save_path="scatter.png",
    axis_min=0, axis_max=10,
    model_name="GNN"
)
```
