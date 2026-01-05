"""
PubChem卤化盐CAS查询

Step 3: 为有效CID的化合物查询卤化盐（Cl, Br, I, F）的CAS号。
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.data.r4n.pubchem_query import add_halide_to_smiles, get_cas_number


def _save_with_checksum(df: pd.DataFrame, filepath: str, metadata: dict) -> str:
    """保存DataFrame并生成校验和"""
    from src.io.integrity import save_checksum
    df.to_csv(filepath, index=False)
    save_checksum(filepath, metadata)
    return filepath


def step3_query_halide_cas(
    input_csv: str,
    output_csv: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Step 3: 为有CID的化合物查询卤化盐CAS号。
    
    Args:
        input_csv: 输入CSV路径（需包含cid列）
        output_csv: 输出CSV路径，默认为 input_csv_with_cas.csv
        verbose: 是否打印进度
        
    Returns:
        str: 输出文件路径
    """
    input_path = Path(input_csv)
    if output_csv is None:
        stem = input_path.stem.replace("_with_props", "").replace("_with_cid", "")
        output_csv = str(input_path.with_name(stem + "_with_cas.csv"))
    
    if verbose:
        print(f"\n[Step 3] Querying halide salt CAS numbers...")
        print(f"  Input: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    if 'cid' not in df.columns:
        raise ValueError("Input CSV must contain 'cid' column. Run Step 1 first.")
    
    # 卤化物映射
    halide_names = {'Cl': 'Chloride', 'Br': 'Bromide', 'I': 'Iodide', 'F': 'Fluoride'}
    
    for name in halide_names.values():
        df[f'{name}_CAS'] = ''
    
    # Filter out rows with NaN or zero CID (invalid compounds)
    df_with_cid = df[(df['cid'].notna()) & (df['cid'] != 0)]
    
    if verbose:
        print(f"  Processing {len(df_with_cid)} compounds with valid CID...")
    
    for idx, row in tqdm(df_with_cid.iterrows(), total=len(df_with_cid), desc="Halide CAS"):
        for halide, name in halide_names.items():
            halide_smiles = add_halide_to_smiles(row['SMILES'], halide)
            cas = get_cas_number(halide_smiles, verbose=False)
            df.at[idx, f'{name}_CAS'] = cas
    
    metadata = {
        "step": "halide_cas_added",
        "total_compounds": len(df),
        "valid_cid_count": int(df['cid'].notna().sum()),
        "has_halide_cas": True
    }
    _save_with_checksum(df, output_csv, metadata)
    
    if verbose:
        print(f"  [OK] Saved to: {output_csv}")
    
    return output_csv
