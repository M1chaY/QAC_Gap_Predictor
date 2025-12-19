# 输入数据处理工作流

该模块用于将包含SMILES和目标值的CSV/Excel文件转换为可直接用于GNN训练的图数据集。

## 功能特性

✅ **支持多种输入格式**

- CSV文件 (`.csv`)
- Excel文件 (`.xlsx`, `.xls`)，支持指定工作表

✅ **智能特征处理**

- 自动检测必需列（SMILES和gap）
- 检测并复用已有的全局分子特征
- 缺失时自动计算全局特征（分子量、可旋转键数、Bertz复杂度）

✅ **可视化进度**

- 使用tqdm显示特征计算和图生成进度

✅ **高效存储**

- 使用joblib格式（压缩）保存
- 支持直接加载到GPU进行训练

---

## 快速开始

### 1. 准备输入文件

输入文件需包含以下必需列：

- `SMILES`: 分子的SMILES字符串
- `gap`: HOMO-LUMO Gap目标值

可选列（如已存在则跳过计算）：

- `mol_weight`: 分子量
- `num_rotatable_bonds`: 可旋转键数量
- `bertz_ct`: Bertz复杂度

**CSV示例** (`data/input.csv`):

```csv
SMILES,gap
CCO,5.23
c1ccccc1,4.56
CC(C)O,5.01
```

**Excel示例** (`data/input.xlsx`):

| SMILES     | gap  | mol_weight |
|------------|------|------------|
| CCO        | 5.23 | 46.07      |
| c1ccccc1   | 4.56 | 78.11      |

### 2. 运行数据处理工作流

**处理CSV文件:**

```bash
python scripts/input_preparation/input_graph_preparation.py \
    --input data/input.csv \
    --output data/prepared_dataset.joblib
```

**处理Excel文件:**

```bash
python scripts/input_preparation/input_graph_preparation.py \
    --input data/input.xlsx \
    --sheet Sheet1 \
    --output data/prepared_dataset.joblib
```

**自定义列名:**

```bash
python scripts/input_preparation/input_graph_preparation.py \
    --input data/input.csv \
    --smiles-col "Molecule" \
    --target-col "HOMO_LUMO_Gap" \
    --output data/prepared_dataset.joblib
```

### 3. 在训练中使用数据集

```python
from pathlib import Path
from torch_geometric.loader import DataLoader
from src.data.graph_dataset_loader import PreparedGraphDataset

# 加载数据集
dataset = PreparedGraphDataset("data/prepared_dataset.joblib")

print(f"样本数量: {len(dataset)}")
print(f"节点特征维度: {dataset.num_node_features}")
print(f"全局特征维度: {dataset.num_global_features}")

# 创建DataLoader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练
for batch in train_loader:
    batch = batch.to(device)
    # ... 训练代码
```

运行完整训练示例:

```bash
python scripts/input_preparation/example_usage.py
```

---

## 命令行参数

### `input_graph_preparation.py`

| 参数             | 必需 | 默认值   | 说明                              |
|------------------|------|----------|-----------------------------------|
| `--input`, `-i`  | ✅   | -        | 输入文件路径                      |
| `--output`, `-o` | ✅   | -        | 输出文件路径（.joblib）           |
| `--sheet`        | ❌   | None     | Excel工作表名（仅Excel文件需要）  |
| `--smiles-col`   | ❌   | `SMILES` | SMILES列名                        |
| `--target-col`   | ❌   | `gap`    | 目标值列名                        |

---

## 工作流步骤说明

### 步骤1: 文件加载与验证

- 自动识别CSV或Excel格式
- Excel文件必须指定工作表名
- 检查文件是否存在

### 步骤2: 必需列验证

检查是否存在:

- `SMILES`列（或自定义列名）
- `gap`列（或自定义列名）

如果缺失，会提示用户并终止程序。

### 步骤3: 数据清理

- 移除目标值缺失的行
- 转换目标值为数值类型
- 输出数据统计信息

### 步骤4: 全局特征计算

检查是否已存在以下列:

- `mol_weight`: 分子量
- `num_rotatable_bonds`: 可旋转键数量
- `bertz_ct`: Bertz复杂度指数

如果**全部存在**，则跳过计算；否则使用RDKit计算缺失特征（带tqdm进度条）。

### 步骤5: SMILES转图特征

使用tqdm进度条可视化:

1. 从SMILES生成3D分子构象
2. 提取节点特征（原子类型、3D坐标、杂化类型、形式电荷）
3. 提取边特征（键类型）
4. 整合全局特征

### 步骤6: 保存数据集

以joblib格式保存，包含:

```python
{
    "graphs": [Data对象列表],
    "metadata": DataFrame(包含SMILES、gap、全局特征),
    "smiles_col": "SMILES",
    "target_col": "gap",
    "num_samples": 样本数量
}
```

---

## 输出文件结构

生成的`.joblib`文件包含:

| 键            | 类型          | 说明                           |
|---------------|---------------|--------------------------------|
| `graphs`      | `List[Data]`  | PyTorch Geometric图对象列表    |
| `metadata`    | `DataFrame`   | 包含SMILES、gap和全局特征      |
| `smiles_col`  | `str`         | SMILES列名                     |
| `target_col`  | `str`         | 目标值列名                     |
| `num_samples` | `int`         | 样本数量                       |

### 图对象 (`Data`) 结构

每个图包含:

```python
Data(
    x=[num_atoms, 9],          # 节点特征: one-hot(C,H,O,N) + 3D坐标 + 电荷 + 杂化
    edge_index=[2, num_edges], # 边索引
    edge_attr=[num_edges, 1],  # 边特征: 键类型
    u=[1, 3],                  # 全局特征: [mol_weight, num_rotatable, bertz_ct]
    y=[1]                      # 目标值: gap
)
```

---

## 为什么选择joblib而非JSON？

| 特性               | joblib     | JSON |
|--------------------|------------|------|
| 存储PyTorch Tensor | ✅         | ❌   |
| 加载速度           | 快         | 慢   |
| 文件大小           | 小（压缩） | 大   |
| 可读性             | 低         | 高   |

joblib专为Python科学计算设计，能直接序列化复杂对象（如`torch.Tensor`），适合深度学习数据集。

---

## 常见问题

### Q1: Excel文件报错"必须指定工作表名"

**A**: 使用`--sheet`参数指定工作表:

```bash
python ... --input data.xlsx --sheet Sheet1
```

### Q2: 提示"输入文件缺少必需的列"

**A**: 确保文件包含`SMILES`和`gap`列，或使用自定义列名:

```bash
python ... --smiles-col "Molecule" --target-col "HOMO_LUMO_Gap"
```

### Q3: 如何查看已有哪些列？

**A**: 错误信息会显示当前文件的所有列名。

### Q4: 数据处理很慢怎么办？

**A**:

- 如果CSV已包含`mol_weight`、`num_rotatable_bonds`、`bertz_ct`，会跳过计算
- 可以预先计算并保存到CSV中加速后续处理

### Q5: 生成的数据集可以直接用于训练吗？

**A**: 是的！使用`PreparedGraphDataset`类可直接加载并封装到`DataLoader`:

```python
from src.data.graph_dataset_loader import PreparedGraphDataset
dataset = PreparedGraphDataset("data/prepared_dataset.joblib")
```

---

## 文件组织

```text
scripts/input_preparation/
├── input_graph_preparation.py   # 主工作流脚本
├── example_usage.py              # 使用示例
└── README.md                     # 本文档

src/data/
├── graph_dataset_loader.py      # 数据集加载工具
└── ...

src/input/
└── smiles_transformation.py     # 核心转换函数（被工作流调用）
```

---

## 下一步

1. **数据准备**: 按要求准备CSV/Excel文件
2. **运行工作流**: 执行`input_graph_preparation.py`
3. **训练模型**: 使用生成的`.joblib`文件进行GNN训练

参考 [example_usage.py](example_usage.py) 了解完整训练流程。
