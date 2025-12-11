"""
GNN数据准备模块

提供分子数据预处理和图数据构建的函数：
1. 预计算分子全局特征（避免重复计算）
2. 将分子数据转换为PyG图神经网络可用的数据格式
3. 批量加载和处理分子图数据
"""

import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    GraphDescriptors,
    rdDistGeom,
    rdForceFieldHelpers,
    rdMolDescriptors,
)
from torch_geometric.data import Data

warnings.filterwarnings('ignore')

__all__ = [
    "preprocess_dataset",
    "load_graph_dataset",
    "mol_to_graph",
    "build_3d_mol",
    "main",
]


_FEATURE_COLUMNS = ["mol_weight", "logp", "num_rotatable_bonds", "bertz_ct"]


def _ensure_valid_columns(df: pd.DataFrame, smiles_col: str, target_col: str) -> None:
    if smiles_col not in df.columns:
        raise ValueError(f"输入文件缺少'{smiles_col}'列")
    if target_col not in df.columns:
        raise ValueError(f"输入文件缺少'{target_col}'列")


def _clean_numeric_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df[df[target_col].notna()].copy()


# ============================================================================
# 分子特征提取
# ============================================================================

def _compute_molecular_descriptors(smiles: str) -> Optional[dict]:
    """从SMILES提取分子级全局描述符。失败返回None。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    mol_no_h = Chem.RemoveHs(mol)

    try:
        return {
            "mol_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "bertz_ct": GraphDescriptors.BertzCT(mol_no_h),
        }
    except Exception as exc:  # pragma: no cover - RDKit errors are data-dependent
        print(f"Error computing descriptors for SMILES '{smiles}': {exc}")
        return None

def preprocess_dataset(
    input_csv: Path,
    output_csv: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap",
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    预处理数据集：计算全局特征并保存
    
    读取包含SMILES和目标值的CSV文件，计算四个全局分子描述符，
    将特征添加到数据集并保存为新的CSV文件。
    
    Args:
        input_csv: 输入CSV文件路径（需包含SMILES和目标值列）
        output_csv: 输出CSV文件路径（将包含全局特征）
        smiles_col: SMILES列名，默认'SMILES'
        target_col: 目标值列名，默认'gap'
        skip_existing: 如果输出文件已存在是否跳过，默认True
        
    Returns:
        处理后的DataFrame
    """
    if skip_existing and output_csv.exists():
        print(f"输出文件已存在: {output_csv}")
        print("加载已有文件...")
        return pd.read_csv(output_csv)

    print(f"读取输入文件: {input_csv}")
    df = pd.read_csv(input_csv)
    _ensure_valid_columns(df, smiles_col, target_col)
    df = _clean_numeric_target(df, target_col)

    print(f"有效样本数: {len(df)}")
    print(f"{target_col}范围: {df[target_col].min():.4f} - {df[target_col].max():.4f}")

    print("计算分子全局特征...")
    features_list = []
    failed_indices = []

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        descriptors = _compute_molecular_descriptors(smiles)

        if descriptors is None:
            failed_indices.append(idx)
            features_list.append({col: None for col in _FEATURE_COLUMNS})
        else:
            features_list.append(descriptors)

    features_df = pd.DataFrame(features_list)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    if failed_indices:
        print(f"警告: {len(failed_indices)}个分子特征计算失败，已移除")
        df = df.dropna(subset=_FEATURE_COLUMNS)

    print(f"成功处理样本数: {len(df)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")

    return df


# ============================================================================
# 3D分子构建和图转换
# ============================================================================

def build_3d_mol(smiles: str) -> Optional[Chem.Mol]:
    """从SMILES生成3D分子结构。外部调用保留为公开接口。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 0xF00D
    embed_result = rdDistGeom.EmbedMolecule(mol, params)
    if embed_result == -1:
        embed_result = rdDistGeom.EmbedMolecule(mol)
        if embed_result == -1:
            return None

    try:
        mmff_result = rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
        if mmff_result == 1:
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
    except ValueError:
        try:
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
        except ValueError:
            pass

    return mol


def _extract_atom_features(atom: Chem.Atom, conf) -> list:
    """提取单个原子的特征。"""
    atom_num = atom.GetAtomicNum()
    atom_type_onehot = [
        1 if atom_num == 6 else 0,  # C
        1 if atom_num == 1 else 0,  # H
        1 if atom_num == 8 else 0,  # O
        1 if atom_num == 7 else 0,  # N
    ]

    pos = conf.GetAtomPosition(atom.GetIdx())
    coords = [pos.x, pos.y, pos.z]

    return atom_type_onehot + coords + [atom.GetFormalCharge(), atom.GetHybridization().real]


def _extract_bond_features(mol: Chem.Mol) -> tuple:
    """提取分子的键特征（边索引和边属性）。"""
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices.append([i, j])
        edge_indices.append([j, i])

        bond_type = bond.GetBondTypeAsDouble()
        edge_attrs.append([bond_type])
        edge_attrs.append([bond_type])

    return edge_indices, edge_attrs


def mol_to_graph(mol: Chem.Mol, gap: float, precomputed_features: Optional[list] = None) -> Optional[Data]:
    """
    将RDKit分子转换为PyG图数据
    
    节点特征：原子类型(one-hot: C,H,O,N)、3D坐标、杂化类型、形式电荷
    边特征：键类型
    全局特征：分子量、LogP等描述符
    
    Args:
        mol: RDKit Mol对象（已加氢的3D构象分子）
        gap: HOMO-LUMO Gap目标值
        precomputed_features: 预计算的全局特征列表 [mol_weight, logp, num_rotatable_bonds, bertz_ct]
                             如果提供，将直接使用而不重新计算
        
    Returns:
        PyG Data对象或None
    """
    if mol is None:
        return None
    
    # 获取全局特征
    if precomputed_features is not None:
        mol_descriptors = precomputed_features
    else:
        mol_no_h = Chem.RemoveHs(mol)
        mol_descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            GraphDescriptors.BertzCT(mol_no_h),
        ]
    
    conf = mol.GetConformer()
    
    # 提取原子特征
    atom_features = [_extract_atom_features(atom, conf) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 提取键特征
    edge_indices, edge_attrs = _extract_bond_features(mol)
    
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    u = torch.tensor([mol_descriptors], dtype=torch.float)
    y = torch.tensor([gap], dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, y=y)
    
    return data


def load_graph_dataset(
    csv_path: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap",
    use_precomputed: bool = True,
) -> List[Data]:
    """
    从CSV文件加载分子图数据集
    
    Args:
        csv_path: CSV文件路径，需包含SMILES和目标值列
        smiles_col: SMILES列名，默认'SMILES'
        target_col: 目标值列名，默认'gap'
        use_precomputed: 是否使用预计算的全局特征，默认True
        
    Returns:
        PyG Data对象列表
    """
    df = pd.read_csv(csv_path)
    _ensure_valid_columns(df, smiles_col, target_col)
    df = _clean_numeric_target(df, target_col)

    print(f"加载了 {len(df)} 个有效样本")
    print(f"{target_col}范围: {df[target_col].min():.2f} - {df[target_col].max():.2f}")

    has_precomputed = all(col in df.columns for col in _FEATURE_COLUMNS)

    if use_precomputed and has_precomputed:
        print("使用预计算的全局特征")
    elif use_precomputed and not has_precomputed:
        print("警告: CSV文件缺少预计算特征，将实时计算")
        print("提示: 先运行 preprocess_dataset() 函数预先计算特征可提高效率")
        has_precomputed = False
    else:
        print("实时计算全局特征")
        has_precomputed = False

    data_list: List[Data] = []
    failed = 0

    for _, row in df.iterrows():
        smiles = row[smiles_col]
        gap = float(row[target_col])

        mol = build_3d_mol(smiles)
        if mol is None:
            failed += 1
            continue

        precomputed_features = None
        if has_precomputed:
            precomputed_features = [float(row[col]) for col in _FEATURE_COLUMNS]

        graph_data = mol_to_graph(mol, gap, precomputed_features)
        if graph_data is not None:
            data_list.append(graph_data)

    print(f"成功创建了 {len(data_list)} 个图对象")
    if failed > 0:
        print(f"处理失败: {failed} 个分子")

    return data_list


# ============================================================================
# 批量预处理脚本
# ============================================================================

def main():
    """
    批量预处理QM9和R4N数据集
    
    用法：
        python -m src.input_procession
    """
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("处理QM9数据集")
    print("=" * 60)
    qm9_input = project_root / "data" / "qm9" / "processed_data" / "qm9_final.csv"
    qm9_output = project_root / "data" / "qm9" / "processed_data" / "qm9_with_features.csv"
    
    if qm9_input.exists():
        preprocess_dataset(
            input_csv=qm9_input,
            output_csv=qm9_output,
            smiles_col='SMILES',
            target_col='gap'
        )
    else:
        print(f"警告: QM9输入文件不存在: {qm9_input}")
    
    print("\n" + "=" * 60)
    print("处理R4N数据集")
    print("=" * 60)
    r4n_input = project_root / "data" / "r4n" / "dataset_r4n_c20.csv"
    r4n_output = project_root / "data" / "r4n" / "dataset_r4n_c20_with_features.csv"
    
    if r4n_input.exists():
        preprocess_dataset(
            input_csv=r4n_input,
            output_csv=r4n_output,
            smiles_col='SMILES',
            target_col='gap'
        )
    else:
        print(f"警告: R4N输入文件不存在: {r4n_input}")
    
    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
