# dataset - 数据集封装模块

提供PyTorch Geometric数据集封装和转换流水线。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `csv_loader.py` | 从CSV加载图数据集 | <150 |
| `joblib_loader.py` | 从joblib加载预处理数据集 | <150 |
| `pipeline.py` | SMILES到图的转换流水线 | <150 |
| `feature_pipeline.py` | 特征计算流水线 | <150 |

## 使用示例

```python
from src.dataset.csv_loader import R4NGapDataset
from torch_geometric.loader import DataLoader

dataset = R4NGapDataset("data/qm9_prepared.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```
