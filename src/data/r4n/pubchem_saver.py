"""
PubChem验证和保存

提供使用PubChem验证化合物并保存结果的功能。
"""

from typing import List

import pandas as pd
from tqdm import tqdm

from src.data.r4n.pubchem_query import (
    validate_pubchem_compound,
    add_halide_to_smiles,
    get_cas_number,
    PUBCHEM_AVAILABLE,
)
from src.data.r4n.statistics import save_compounds_to_csv


def validate_with_pubchem(
    df: pd.DataFrame, 
    get_properties: bool = False, 
    get_halide_cas: bool = False, 
    verbose: bool = True
) -> pd.DataFrame:
    """
    使用PubChem验证化合物。
    
    Args:
        df: 包含SMILES的DataFrame
        get_properties: 是否获取分子属性
        get_halide_cas: 是否查询卤化盐CAS号
        verbose: 是否打印进度
        
    Returns:
        pd.DataFrame: 添加了PubChem信息的DataFrame
    """
    if not PUBCHEM_AVAILABLE:
        raise ImportError("pubchempy not installed")
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="PubChem Validation"):
        result = validate_pubchem_compound(
            row['SMILES'], verbose=False, get_properties=get_properties
        )
        results.append(result if result else {})
    
    results_df = pd.DataFrame(results)
    df = pd.concat([df, results_df], axis=1)
    
    if verbose:
        found = df['cid'].notna().sum()
        print(f"\nFound in PubChem: {found}/{len(df)}")
    
    if get_halide_cas:
        df = query_halide_cas(df, verbose)
    
    return df


def query_halide_cas(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    查询卤化盐CAS号。
    
    Args:
        df: 包含SMILES和cid的DataFrame
        verbose: 是否打印进度
        
    Returns:
        pd.DataFrame: 添加了卤化盐CAS号的DataFrame
    """
    halides = ['Cl', 'Br', 'I', 'F']
    halide_names = {'Cl': 'Chloride', 'Br': 'Bromide', 'I': 'Iodide', 'F': 'Fluoride'}
    
    for halide in halides:
        df[f'{halide_names[halide]}_CAS'] = ''
    
    df_with_cid = df[df['cid'].notna()]
    
    for idx, row in tqdm(df_with_cid.iterrows(), total=len(df_with_cid), desc="Halide CAS"):
        for halide in halides:
            halide_smiles = add_halide_to_smiles(row['SMILES'], halide)
            cas = get_cas_number(halide_smiles, verbose=False)
            df.at[idx, f'{halide_names[halide]}_CAS'] = cas
    
    return df


def save_with_validation(
    compounds: List,
    filename: str,
    max_carbons: int,
    validate_pubchem: bool = False,
    get_properties: bool = False,
    get_halide_cas: bool = False,
    verbose: bool = True
) -> str:
    """
    保存化合物到CSV并可选验证。
    
    Args:
        compounds: 化合物列表
        filename: 输出文件名
        max_carbons: 最大碳原子数
        validate_pubchem: 是否使用PubChem验证
        get_properties: 是否获取分子属性
        get_halide_cas: 是否查询卤化盐CAS
        verbose: 是否打印进度
        
    Returns:
        str: 保存的文件路径
    """
    if filename is None:
        filename = f"data/r4n_smiles_c{max_carbons}.csv"

    total = len(compounds)
    index_width = max(1, len(str(total)))

    data = []
    for i, (carbon_count, smiles) in enumerate(compounds, 1):
        data.append({
            'Index': f"{i:0{index_width}d}",
            'Num_c': carbon_count,
            'SMILES': smiles
        })
    
    df = pd.DataFrame(data)
    
    if validate_pubchem:
        df = validate_with_pubchem(df, get_properties, get_halide_cas, verbose)
    
    df.to_csv(filename, index=False)
    
    if verbose:
        print(f"\nResult saved to {filename}")
    
    return filename
