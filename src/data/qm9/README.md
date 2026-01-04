# QM9 数据集处理模块

提供 QM9 数据集的加载、提取、过滤和预处理功能。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `loader.py` | 数据集检查和加载 | <150 |
| `extractor.py` | 分子信息提取 | <150 |
| `atom_filter.py` | 原子类型过滤（氟等） | <150 |
| `smiles_converter.py` | SMILES标准化转换 | <150 |
| `preprocessor.py` | 预处理和特征计算 | <150 |
| `pipeline.py` | 完整处理流水线 | <150 |

## 使用示例

```python
from src.data.qm9.pipeline import extract_qm9
from pathlib import Path

df = extract_qm9(Path("data/qm9"))
```
