# io - 输入输出模块

提供文件加载、数据验证和数据集保存功能。

## 文件说明

| 文件 | 功能 | 行数限制 |
| ---- | ---- | -------- |
| `file_loader.py` | CSV/Excel文件加载 | <150 |
| `validator.py` | 数据验证和清理 | <150 |
| `saver.py` | 数据集保存 | <150 |
| `integrity.py` | 文件校验和完整性检查 | <150 |
| `report_saver.py` | Optuna搜索结果保存 | <150 |
| `training_logger.py` | 训练日志记录工具 | <150 |

## 使用示例

### 文件加载和验证

```python
from src.io import load_input_file, validate_required_columns, clean_target_data

df = load_input_file("data/input.csv")
validate_required_columns(df)
df = clean_target_data(df)
```

### 保存Optuna搜索结果

```python
from src.io import save_optuna_results

results = {
    "search_info": {...},
    "dataset_info": {...},
    "best_params": {...},
    "best_performance": {...},
    "study_statistics": {...}
}

json_path, md_path = save_optuna_results(results, "models/")
```

### 训练日志记录

```python
from src.io import (
    setup_training_logger, 
    log_trial_start, 
    log_epoch, 
    log_trial_end,
    log_search_summary
)

# 创建日志记录器
logger = setup_training_logger("models/", log_name="optuna_search")

# 记录trial开始
params = {'hidden_channels': 64, 'num_heads': 4, 'lr': 0.001}
log_trial_start(logger, trial_number=1, params=params)

# 记录每个epoch
log_epoch(logger, trial_number=1, epoch=1, train_loss=0.5, 
          val_mae=0.3, val_r2=0.8, lr=0.001, is_best=True)

# 记录trial结束
log_trial_end(logger, trial_number=1, best_val_mae=0.25, 
              total_epochs=100, status="COMPLETED")
```
