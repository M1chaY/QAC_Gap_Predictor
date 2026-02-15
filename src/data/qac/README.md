# QAC 数据集处理模块

提供季铵离子 (QAC) 化合物的生成、验证和清洗功能。

## 文件说明

| 文件 | 功能 |
| ---- | ---- |
| `smiles_builder.py` | 季铵离子SMILES构建 |
| `alkyl_groups.py` | 烷基基团生成 |
| `generator.py` | 化合物生成器类 |
| `molecule_validator.py` | 分子结构验证 |
| `pubchem_query.py` | PubChem查询功能 |
| `step1_cid.py` | Step 1: CID验证 |
| `step2_properties.py` | Step 2: 属性查询 |
| `step3_halide_cas.py` | Step 3: 卤化盐CAS查询 |
| `pubchem_pipeline.py` | 完整验证流水线 |
| `statistics.py` | 统计信息和CSV保存 |
| `cleaner.py` | 数据集清洗 |

## 使用示例

### 生成化合物

```python
from src.data.qac import QACGenerator, print_statistics

generator = QACGenerator(max_carbons=20)
compounds, distribution = generator.generate_compounds()
print_statistics(compounds, distribution)
```

### 保存并验证

```python
from src.data.qac import save_compounds_to_csv
from src import run_full_validation_pipeline

# 保存基础数据
save_compounds_to_csv(compounds, "data/qac/output.csv", max_carbons=20)

# 运行完整验证流水线
run_full_validation_pipeline("data/qac/output.csv", verbose=True)
```

### 分步验证

```python
from src import step1_validate_cid, step2_add_properties, step3_query_halide_cas

# 每步独立运行，支持断点续传
step1_validate_cid("input.csv", "output_with_cid.csv")
step2_add_properties("output_with_cid.csv", "output_with_props.csv")
step3_query_halide_cas("output_with_props.csv", "output_with_cas.csv")
```
