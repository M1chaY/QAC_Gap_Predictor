"""
数据预处理工具函数

用于处理输入文件并转换为图数据集的核心函数
"""

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors
from torch_geometric.data import Data
from tqdm import tqdm

from src.input.smiles_transformation import build_3d_mol, mol_to_graph


def load_input_file(
    file_path: Path, 
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    加载输入文件（CSV或Excel）
    
    Args:
        file_path: 输入文件路径
        sheet_name: Excel工作表名（仅对Excel文件有效）
        
    Returns:
        DataFrame
        
    Raises:
        ValueError: 如果文件格式不支持或Excel文件未指定工作表
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".csv":
        print(f"读取CSV文件: {file_path}")
        return pd.read_csv(file_path)
    
    elif suffix in [".xlsx", ".xls"]:
        if sheet_name is None:
            raise ValueError("Excel文件必须指定工作表名")
        print(f"读取Excel文件: {file_path}, 工作表: {sheet_name}")
        return pd.read_excel(file_path, sheet_name=sheet_name)
    
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，仅支持.csv、.xlsx、.xls")


def validate_required_columns(
    df: pd.DataFrame, 
    smiles_col: str = "SMILES", 
    target_col: str = "gap"
) -> None:
    """
    验证DataFrame中是否存在必需的列
    
    Args:
        df: 输入DataFrame
        smiles_col: SMILES列名
        target_col: 目标值列名
        
    Raises:
        ValueError: 如果缺少必需列
    """
    missing_cols = []
    
    if smiles_col not in df.columns:
        missing_cols.append(smiles_col)
    if target_col not in df.columns:
        missing_cols.append(target_col)
    
    if missing_cols:
        raise ValueError(
            f"输入文件缺少必需的列: {', '.join(missing_cols)}\n"
            f"当前文件列: {', '.join(df.columns)}\n"
            f"请确保文件包含'{smiles_col}'列（分子SMILES）和'{target_col}'列（目标值）"
        )
    
    print(f"✓ 必需列检查通过: {smiles_col}, {target_col}")


def clean_data(
    df: pd.DataFrame, 
    target_col: str = "gap"
) -> pd.DataFrame:
    """
    清理数据：移除缺失值和无效目标值
    
    Args:
        df: 输入DataFrame
        target_col: 目标值列名
        
    Returns:
        清理后的DataFrame
    """
    original_len = len(df)
    
    # 移除目标值缺失的行
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()
    
    cleaned_len = len(df)
    removed = original_len - cleaned_len
    
    if removed > 0:
        print(f"⚠ 移除了 {removed} 行缺失或无效的目标值")
    
    print(f"✓ 数据清理完成，剩余 {cleaned_len} 个有效样本")
    print(f"  {target_col}范围: {df[target_col].min():.4f} - {df[target_col].max():.4f}")
    
    return df


def compute_global_features(
    df: pd.DataFrame, 
    smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    计算全局分子特征（如果不存在）
    
    Args:
        df: 输入DataFrame
        smiles_col: SMILES列名
        
    Returns:
        添加了全局特征的DataFrame
    """
    feature_cols = ["mol_weight", "num_rotatable_bonds", "bertz_ct"]
    
    # 检查是否已存在全局特征
    has_all_features = all(col in df.columns for col in feature_cols)
    
    if has_all_features:
        print("✓ 检测到已存在全局特征，跳过计算")
        return df
    
    print("⚙ 开始计算全局分子特征...")
    
    mol_weights = []
    num_rotatable = []
    bertz_cts = []
    
    for smiles in tqdm(df[smiles_col], desc="计算全局特征"):
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            mol_weights.append(None)
            num_rotatable.append(None)
            bertz_cts.append(None)
            continue
        
        mol_no_h = Chem.RemoveHs(mol)
        
        mol_weights.append(Descriptors.MolWt(mol))
        num_rotatable.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        bertz_cts.append(GraphDescriptors.BertzCT(mol_no_h))
    
    df["mol_weight"] = mol_weights
    df["num_rotatable_bonds"] = num_rotatable
    df["bertz_ct"] = bertz_cts
    
    # 移除特征计算失败的行
    original_len = len(df)
    df = df.dropna(subset=feature_cols).copy()
    removed = original_len - len(df)
    
    if removed > 0:
        print(f"⚠ 移除了 {removed} 行特征计算失败的样本")
    
    print(f"✓ 全局特征计算完成")
    
    return df


def convert_smiles_to_graphs(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "gap"
) -> Tuple[List[Data], pd.DataFrame]:
    """
    将SMILES转换为图特征
    
    Args:
        df: 包含SMILES和特征的DataFrame
        smiles_col: SMILES列名
        target_col: 目标值列名
        
    Returns:
        图数据列表和更新后的DataFrame（移除了转换失败的行）
    """
    print("⚙ 开始将SMILES转换为图特征...")
    
    feature_cols = ["mol_weight", "num_rotatable_bonds", "bertz_ct"]
    graph_list = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="生成图特征"):
        smiles = row[smiles_col]
        gap = float(row[target_col])
        
        # 构建3D分子
        mol = build_3d_mol(smiles)
        if mol is None:
            continue
        
        # 提取预计算的全局特征
        precomputed_features = [float(row[col]) for col in feature_cols]
        
        # 转换为图
        graph_data = mol_to_graph(mol, gap, precomputed_features)
        if graph_data is None:
            continue
        
        graph_list.append(graph_data)
        valid_indices.append(idx)
    
    # 更新DataFrame，只保留成功转换的行
    df_valid = df.loc[valid_indices].copy()
    
    failed = len(df) - len(graph_list)
    if failed > 0:
        print(f"⚠ {failed} 个分子转换失败，已跳过")
    
    print(f"✓ 成功生成 {len(graph_list)} 个图对象")
    
    return graph_list, df_valid


def save_dataset(
    graph_list: List[Data],
    df: pd.DataFrame,
    output_path: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap"
) -> None:
    """
    保存数据集为joblib格式
    
    保存内容:
    - graphs: PyG Data对象列表
    - metadata: DataFrame（包含SMILES、gap和全局特征）
    
    Args:
        graph_list: 图数据列表
        df: DataFrame元数据
        output_path: 输出文件路径
        smiles_col: SMILES列名
        target_col: 目标值列名
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备保存的数据
    dataset = {
        "graphs": graph_list,
        "metadata": df,
        "smiles_col": smiles_col,
        "target_col": target_col,
        "num_samples": len(graph_list)
    }
    
    print(f"💾 保存数据集到: {output_path}")
    joblib.dump(dataset, output_path, compress=3)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ 保存完成！文件大小: {file_size_mb:.2f} MB")
    print(f"  - 图数据: {len(graph_list)} 个")
    print(f"  - 元数据: {len(df)} 行 × {len(df.columns)} 列")
