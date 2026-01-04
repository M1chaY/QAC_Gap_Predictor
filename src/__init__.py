"""
R4N Gap Predictor - 季铵离子 HOMO-LUMO Gap 预测包

主要功能模块:
- data/qm9: QM9 数据集处理
- data/r4n: R4N 数据集生成
- molecule: 分子构建与图转换
- io: 文件读写与数据验证
- dataset: 数据集封装与转换
- model: GNN 模型定义与训练
"""

# ==================== QM9 数据处理 ====================
from src.data.qm9.loader import check_qm9_data, load_qm9_dataset
from src.data.qm9.extractor import extract_qm9_all_info
from src.data.qm9.atom_filter import filter_fluorine, filter_stereochemistry
from src.data.qm9.smiles_converter import convert_to_standard_smiles
from src.data.qm9.preprocessor import (
    compute_molecular_descriptors,
    preprocess_qm9_dataset,
)
from src.data.qm9.pipeline import extract_qm9

# ==================== R4N 数据处理 ====================
from src.data.r4n.generator import R4NGenerator
from src.data.r4n.smiles_builder import build_r4n_smiles
from src.data.r4n.alkyl_groups import generate_alkyl_groups
from src.data.r4n.molecule_validator import validate_r4n_molecule, get_canonical_smiles
from src.data.r4n.pubchem_query import (
    validate_pubchem_compound,
    add_halide_to_smiles,
    get_cas_number,
)
from src.data.r4n.cleaner import clean_r4n_dataset

# ==================== 分子处理 ====================
from src.molecule.builder import build_3d_mol
from src.molecule.graph_converter import (
    extract_atom_features,
    extract_bond_features,
    mol_to_graph,
)
from src.molecule.features import (
    compute_global_descriptors,
    FEATURE_COLUMNS,
)
from src.molecule.structure_generator import generate_structure

# ==================== 文件IO ====================
from src.io.file_loader import load_input_file
from src.io.validator import validate_required_columns, clean_target_data
from src.io.saver import save_graph_dataset

# ==================== 数据集封装 ====================
from src.dataset.csv_loader import load_graph_dataset, R4NGapDataset
from src.dataset.joblib_loader import load_prepared_dataset, PreparedGraphDataset
from src.dataset.pipeline import convert_smiles_to_graphs
from src.dataset.feature_pipeline import compute_global_features

# ==================== 模型 ====================
from src.model.gap_gnn import GapPredictionGNN
from src.model.training import train_epoch, evaluate, compute_loss
from src.model.visualization import plot_loss_curves, plot_prediction_scatter
from src.model.evaluation import calculate_metrics, metrics_to_dataframe
from src.model.scatter_plot import plot_actual_vs_predicted

# ==================== 导出列表 ====================
__all__ = [
    # QM9 数据处理
    "check_qm9_data",
    "load_qm9_dataset",
    "extract_qm9_all_info",
    "filter_fluorine",
    "filter_stereochemistry",
    "convert_to_standard_smiles",
    "compute_molecular_descriptors",
    "preprocess_qm9_dataset",
    "extract_qm9",
    # R4N 数据处理
    "R4NGenerator",
    "build_r4n_smiles",
    "generate_alkyl_groups",
    "validate_r4n_molecule",
    "get_canonical_smiles",
    "validate_pubchem_compound",
    "add_halide_to_smiles",
    "get_cas_number",
    "clean_r4n_dataset",
    # 分子处理
    "build_3d_mol",
    "extract_atom_features",
    "extract_bond_features",
    "mol_to_graph",
    "compute_global_descriptors",
    "FEATURE_COLUMNS",
    "generate_structure",
    # 文件IO
    "load_input_file",
    "validate_required_columns",
    "clean_target_data",
    "save_graph_dataset",
    # 数据集封装
    "load_graph_dataset",
    "R4NGapDataset",
    "load_prepared_dataset",
    "PreparedGraphDataset",
    "convert_smiles_to_graphs",
    "compute_global_features",
    # GNN 模型
    "GapPredictionGNN",
    "train_epoch",
    "evaluate",
    "compute_loss",
    "plot_loss_curves",
    "plot_prediction_scatter",
    # 模型评估
    "calculate_metrics",
    "metrics_to_dataframe",
    "plot_actual_vs_predicted",
]