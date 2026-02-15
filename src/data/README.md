# data - 数据处理模块

本目录包含 QM9 和 QAC 数据集的加载、处理和生成功能。

## 目录结构

```txt
data/
├── qm9/           # QM9 数据集处理
└── qac/           # QAC 数据集生成
```

## 子模块说明

### qm9/ - QM9 数据集处理

| 文件 | 功能 |
|------|------|
| `loader.py` | QM9 数据集检查和加载 |
| `extractor.py` | QM9 分子信息提取 |
| `atom_filter.py` | 氟原子和立体化学过滤 |
| `smiles_converter.py` | SMILES 标准化转换 |
| `preprocessor.py` | 预处理和分子描述符计算 |
| `pipeline.py` | QM9 完整处理流水线 |

### qac/ - QAC 数据集生成

| 文件 | 功能 |
|------|------|
| `generator.py` | 季铵离子化合物生成器 |
| `smiles_builder.py` | QAC SMILES 字符串构建 |
| `alkyl_groups.py` | 烷基取代基生成 |
| `molecule_validator.py` | 分子结构验证 |
| `pubchem_query.py` | PubChem 数据库查询 |
| `cleaner.py` | QAC 数据集清洗 |

## 使用示例

### 提取 QM9 数据集

```python
from pathlib import Path
from src.data.qm9 import extract_qm9

qm9_dir = Path("data/qm9")
df = extract_qm9(qm9_dir)
```

### 生成 QAC 化合物

```python
from src.data.qac import QACGenerator

generator = QACGenerator(max_carbons=20)
compounds, distribution = generator.generate_compounds()
```
