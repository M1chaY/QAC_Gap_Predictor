"""
QM9数据预处理器

提供分子描述符计算功能。
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors


FEATURE_COLUMNS = ["mol_weight", "num_rotatable_bonds", "bertz_ct"]


def compute_molecular_descriptors(smiles: str) -> Optional[dict]:
    """
    从SMILES提取分子级全局描述符。
    
    Args:
        smiles: 分子的SMILES字符串
        
    Returns:
        dict: 包含mol_weight, num_rotatable_bonds, bertz_ct，失败返回None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    mol_no_h = Chem.RemoveHs(mol)

    try:
        return {
            "mol_weight": Descriptors.MolWt(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "bertz_ct": GraphDescriptors.BertzCT(mol_no_h),
        }
    except Exception as exc:
        print(f"Error computing descriptors for SMILES '{smiles}': {exc}")
        return None


def preprocess_qm9_dataset(
    input_csv: Path,
    output_csv: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap",
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    预处理QM9数据集：计算全局特征并保存。
    
    Args:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        smiles_col: SMILES列名
        target_col: 目标值列名
        skip_existing: 输出文件已存在时是否跳过
        
    Returns:
        pd.DataFrame: 处理后的DataFrame
    """
    if skip_existing and output_csv.exists():
        print(f"Output file exists: {output_csv}")
        return pd.read_csv(output_csv)

    print(f"Reading input file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if smiles_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Missing required columns: {smiles_col}, {target_col}")
    
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    print(f"Valid samples: {len(df)}")

    print("Computing molecular global features...")
    features_list = []
    failed_indices = []

    for idx, row in df.iterrows():
        descriptors = compute_molecular_descriptors(row[smiles_col])
        if descriptors is None:
            failed_indices.append(idx)
            features_list.append({col: None for col in FEATURE_COLUMNS})
        else:
            features_list.append(descriptors)

    features_df = pd.DataFrame(features_list)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    if failed_indices:
        print(f"Warning: {len(failed_indices)} molecules failed, removed")
        df = df.dropna(subset=FEATURE_COLUMNS)

    print(f"Successfully processed: {len(df)} samples")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to: {output_csv}")

    return df
