"""
QAC数据预处理器

提供QAC数据集预处理功能：合并SMILES与gap值，计算分子描述符。
"""

from pathlib import Path

import pandas as pd

from src.data.qm9.preprocessor import compute_molecular_descriptors, FEATURE_COLUMNS
from src.io.integrity import save_checksum
from src.path import DATA_DIR, QAC_DIR


def preprocess_qac_dataset(
    smiles_csv: Path,
    gap_xlsx: Path,
    output_csv: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap",
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    预处理QAC数据集：合并SMILES与gap值，计算全局特征并保存。
    
    Args:
        smiles_csv: SMILES数据CSV文件路径
        gap_xlsx: gap数据Excel文件路径
        output_csv: 输出CSV文件路径
        smiles_col: SMILES列名
        target_col: 目标值列名
        skip_existing: 输出文件已存在时是否跳过
        
    Returns:
        pd.DataFrame: 处理后的DataFrame
    """
    from src.io.integrity import check_data_integrity
    
    if skip_existing and output_csv.exists():
        if check_data_integrity(str(output_csv), verbose=True):
            print("Loading existing valid data...")
            return pd.read_csv(output_csv)
        else:
            print("Existing data failed integrity check. Reprocessing...")
    
    # 1. 加载SMILES数据
    print(f"Reading SMILES data: {smiles_csv}")
    smiles_df = pd.read_csv(smiles_csv)
    print(f"  Loaded {len(smiles_df)} records")
    smiles_df = smiles_df[[smiles_col]].copy()
    
    # 2. 加载gap数据
    print(f"Reading gap data: {gap_xlsx}")
    gap_df = pd.read_excel(gap_xlsx)
    print(f"  Loaded {len(gap_df)} records")
    
    # 3. 合并数据
    print("Merging data by SMILES...")
    if smiles_col not in gap_df.columns or target_col not in gap_df.columns:
        raise ValueError(f"gap_df must contain {smiles_col} and {target_col} columns")
    
    df = smiles_df.merge(
        gap_df[[smiles_col, target_col]], 
        on=smiles_col, 
        how="left"
    )
    print(f"  Merged records: {len(df)}")
    
    # 4. 清理gap数据
    print("Cleaning gap data...")
    original_len = len(df)
    
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()
    df = df[df[target_col] != 0].copy()
    
    removed = original_len - len(df)
    print(f"  Removed {removed} rows with missing/zero gap")
    print(f"  Valid samples: {len(df)}")
    
    # 5. 计算全局特征
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
        print(f"  Warning: {len(failed_indices)} molecules failed, removed")
        df = df.dropna(subset=FEATURE_COLUMNS)
    
    print(f"  Successfully processed: {len(df)} samples")
    
    # 6. 保存结果
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    metadata = {
        "type": "qac_preprocessed",
        "total_samples": len(df),
        "smiles_col": smiles_col,
        "target_col": target_col
    }
    save_checksum(str(output_csv), metadata)
    
    print(f"Results saved to: {output_csv}")
    
    return df
