"""
R4N Gap Predictor - 季铵离子 HOMO-LUMO Gap 预测包

主要功能模块:
- data: 数据集生成与处理 (QM9, R4N)
- input: 输入数据预处理与图转换
- model: GNN 模型定义与训练
"""

# ==================== 数据处理模块 ====================
# QM9 数据集处理
from .data.data_qm9 import extract_qm9, preprocess_dataset

# R4N 数据集生成与清洗
from .data.data_r4n import R4NGenerator, clean_r4n_data

# Mol 文件结构生成
from .data.molfile_r4n_structure import generate_structure

# 图数据集加载
from .data.graph_dataset_loader import load_prepared_dataset, PreparedGraphDataset

# ==================== 输入预处理模块 ====================
# SMILES 转换与图构建
from .input.smiles_transformation import (
    build_3d_mol,
    mol_to_graph,
    load_graph_dataset,
    R4NGapDataset,
)

# 通用预处理工具
from .input.preprocessing import (
    load_input_file,
    validate_required_columns,
    clean_data,
    compute_global_features,
    convert_smiles_to_graphs,
    save_dataset,
)

# ==================== 模型模块 ====================
# GNN 模型与训练
from .model.gnn import GapPredictionGNN, train_epoch, evaluate, compute_loss, plot_loss_curves

# ==================== 导出列表 ====================
__all__ = [
    # 数据处理
    "extract_qm9",
    "preprocess_dataset",
    "R4NGenerator",
    "clean_r4n_data",
    "generate_structure",
    "load_prepared_dataset",
    "PreparedGraphDataset",
    # 输入预处理
    "build_3d_mol",
    "mol_to_graph",
    "load_graph_dataset",
    "R4NGapDataset",
    "load_input_file",
    "validate_required_columns",
    "clean_data",
    "compute_global_features",
    "convert_smiles_to_graphs",
    "save_dataset",
    # 模型
    "GapPredictionGNN",
    "train_epoch",
    "evaluate",
    "compute_loss",
    "plot_loss_curves",
]