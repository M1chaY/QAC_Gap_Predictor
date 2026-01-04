# io - 输入输出模块

提供文件加载、数据验证和数据集保存功能。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `file_loader.py` | CSV/Excel文件加载 | <150 |
| `validator.py` | 数据验证和清理 | <150 |
| `saver.py` | 数据集保存 | <150 |

## 使用示例

```python
from src.io.file_loader import load_input_file
from src.io.validator import validate_required_columns, clean_target_data
from src.io.saver import save_graph_dataset

df = load_input_file("data/input.csv")
validate_required_columns(df)
df = clean_target_data(df)
```
