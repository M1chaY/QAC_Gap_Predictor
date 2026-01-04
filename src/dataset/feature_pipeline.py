"""
特征计算流水线

批量计算分子全局特征。
"""

import pandas as pd
from tqdm import tqdm

from src.molecule.features import compute_features_from_smiles, FEATURE_COLUMNS


def compute_global_features(
    df: pd.DataFrame, 
    smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    计算全局分子特征（如果不存在）。
    
    计算的特征包括：
    - mol_weight: 分子量
    - num_rotatable_bonds: 可旋转键数量
    - bertz_ct: Bertz复杂度指数
    
    Args:
        df: 输入DataFrame
        smiles_col: SMILES列名
        
    Returns:
        pd.DataFrame: 添加了全局特征的DataFrame
    """
    has_all_features = all(col in df.columns for col in FEATURE_COLUMNS)
    
    if has_all_features:
        print("Global features already exist, skipping computation")
        return df
    
    print("Computing global molecular features...")
    
    mol_weights = []
    num_rotatable = []
    bertz_cts = []
    
    for smiles in tqdm(df[smiles_col], desc="Computing features"):
        features = compute_features_from_smiles(smiles)
        
        if features is None:
            mol_weights.append(None)
            num_rotatable.append(None)
            bertz_cts.append(None)
        else:
            mol_weights.append(features["mol_weight"])
            num_rotatable.append(features["num_rotatable_bonds"])
            bertz_cts.append(features["bertz_ct"])
    
    df["mol_weight"] = mol_weights
    df["num_rotatable_bonds"] = num_rotatable
    df["bertz_ct"] = bertz_cts
    
    original_len = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS).copy()
    removed = original_len - len(df)
    
    if removed > 0:
        print(f"Removed {removed} rows with failed feature computation")
    
    print("Global feature computation complete")
    
    return df
