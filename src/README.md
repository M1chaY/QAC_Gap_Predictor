# src - 源代码目录

本目录包含 QAC Gap Predictor 项目的核心源代码。

## 目录结构

```text
src/
├── __init__.py          # 包初始化，导出公共API
├── path.py              # 项目路径配置
├── data/                # 数据处理模块
│   ├── qm9/             # QM9 数据集处理
│   └── qac/             # QAC 数据集生成
├── molecule/            # 分子处理模块
├── io/                  # 文件IO模块
├── dataset/             # 数据集封装模块
└── model/               # 模型定义模块
```

## 模块说明

| 模块 | 功能 |
| ---- | ---- |
| `data/qm9/` | QM9 数据集加载、提取、过滤和预处理 |
| `data/qac/` | QAC 季铵离子化合物生成和验证 |
| `molecule/` | 3D 分子构建、图转换、特征计算 |
| `io/` | 文件加载、数据验证、数据集保存 |
| `dataset/` | PyTorch Geometric 数据集封装 |
| `model/` | GNN 模型定义、训练工具、可视化 |

## 使用示例

```python
from src import (
    # QM9 数据处理
    extract_qm9,
    # QAC 数据生成
    QACGenerator,
    # 分子处理
    build_3d_mol,
    mol_to_graph,
    # 数据集
    QACGapDataset,
    PreparedGraphDataset,
    # 模型
    GapPredictionGNN,
    train_epoch,
    evaluate,
)
```

## 代码规范

- 每个 Python 文件专注于单一功能
- 每个文件不超过 150 行代码
- 遵循 PEP 8 代码风格
- 所有函数和类都有完整的文档字符串
