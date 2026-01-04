"""
数据验证器

提供DataFrame数据验证和清理功能。
"""

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame, 
    smiles_col: str = "SMILES", 
    target_col: str = "gap"
) -> None:
    """
    验证DataFrame中是否存在必需的列。
    
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
            f"Missing required columns: {', '.join(missing_cols)}\n"
            f"Current columns: {', '.join(df.columns)}"
        )
    
    print(f"Required columns check passed: {smiles_col}, {target_col}")


def clean_target_data(
    df: pd.DataFrame, 
    target_col: str = "gap"
) -> pd.DataFrame:
    """
    清理目标值数据：移除缺失值和无效值。
    
    Args:
        df: 输入DataFrame
        target_col: 目标值列名
        
    Returns:
        pd.DataFrame: 清理后的DataFrame
    """
    original_len = len(df)
    
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()
    
    cleaned_len = len(df)
    removed = original_len - cleaned_len
    
    if removed > 0:
        print(f"Removed {removed} rows with missing/invalid target values")
    
    print(f"Data cleaning complete, {cleaned_len} valid samples")
    print(f"  {target_col} range: {df[target_col].min():.4f} - {df[target_col].max():.4f}")
    
    return df
