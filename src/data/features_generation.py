import pandas as pd
from pathlib import Path
from typing import Optional
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    GraphDescriptors,
    rdMolDescriptors,
)

_FEATURE_COLUMNS = ["mol_weight", "num_rotatable_bonds", "bertz_ct"]


def _ensure_valid_columns(df: pd.DataFrame, smiles_col: str, target_col: str) -> None:
    if smiles_col not in df.columns:
        raise ValueError(f"输入文件缺少'{smiles_col}'列")
    if target_col not in df.columns:
        raise ValueError(f"输入文件缺少'{target_col}'列")


def _clean_numeric_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df[df[target_col].notna()].copy()


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
    
    读取包含SMILES和目标值的CSV文件, 计算四个全局分子描述符, 
    将特征添加到数据集并保存为新的CSV文件。
    
    Args:
        input_csv: 输入CSV文件路径（需包含SMILES和目标值列）
        output_csv: 输出CSV文件路径（将包含全局特征）
        smiles_col: SMILES列名, 默认'SMILES'
        target_col: 目标值列名, 默认'gap'
        skip_existing: 如果输出文件已存在是否跳过, 默认True
        
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
        print(f"警告: {len(failed_indices)}个分子特征计算失败, 已移除")
        df = df.dropna(subset=_FEATURE_COLUMNS)

    print(f"成功处理样本数: {len(df)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")

    return df