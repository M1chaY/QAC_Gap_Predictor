# R4N 数据集处理模块

提供季铵离子 (R4N+) 化合物的生成、验证和清洗功能。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `smiles_builder.py` | 季铵离子SMILES构建 | <150 |
| `alkyl_groups.py` | 烷基基团生成 | <150 |
| `generator.py` | 化合物生成器类 | <150 |
| `molecule_validator.py` | 分子结构验证 | <150 |
| `pubchem_query.py` | PubChem查询功能 | <150 |
| `pubchem_saver.py` | PubChem验证和保存 | <150 |
| `statistics.py` | 统计信息和CSV保存 | <150 |
| `cleaner.py` | 数据集清洗 | <150 |

## 使用示例

### 生成化合物

```python
from src.data.r4n import R4NGenerator, print_statistics

generator = R4NGenerator(max_carbons=20)
compounds, distribution = generator.generate_compounds()
print_statistics(compounds, distribution)
```

### 保存结果

```python
from src.data.r4n import save_compounds_to_csv, save_with_validation

# 简单保存
save_compounds_to_csv(compounds, "output.csv", max_carbons=20)

# 带PubChem验证
save_with_validation(compounds, "output.csv", max_carbons=20, validate_pubchem=True)
```
