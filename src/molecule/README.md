# molecule - 分子处理模块

提供分子的3D构建、图转换和特征计算功能。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `builder.py` | 从SMILES构建3D分子 | <150 |
| `graph_converter.py` | 分子转PyG图数据 | <150 |
| `features.py` | 全局分子特征计算 | <150 |
| `structure_generator.py` | MOL文件生成 | <150 |

## 使用示例

```python
from src.molecule.builder import build_3d_mol
from src.molecule.graph_converter import mol_to_graph

mol = build_3d_mol("CCO")
graph = mol_to_graph(mol, gap=5.0)
```
