# dataset - 数据集封装模块

提供PyTorch Geometric数据集封装和转换流水线。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `csv_loader.py` | 从CSV加载图数据集 | <150 |
| `joblib_loader.py` | 从joblib加载预处理数据集 | <150 |
| `pipeline.py` | SMILES到图的转换流水线 | <150 |
| `feature_pipeline.py` | 特征计算流水线 | <150 |
| `splitter.py` | 数据集划分和随机种子工具 | <150 |

## 使用示例

### 加载数据集

```python
from src.dataset import QACGapDataset, PreparedGraphDataset
from torch_geometric.loader import DataLoader

# 从CSV加载
dataset = QACGapDataset("data/qm9_prepared.csv")

# 从joblib加载
dataset = PreparedGraphDataset("data/qm9_prepared.joblib")

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 划分数据集

```python
from src.dataset import split_dataset, set_seed

# 设置随机种子
set_seed(42)

# 按70:15:15划分
train, val, test = split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42)
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```
