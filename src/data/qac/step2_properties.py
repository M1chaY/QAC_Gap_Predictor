"""
PubChem属性查询

Step 2: 为有效CID的化合物获取额外属性（分子量、IUPAC名等）。
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.data.qac.pubchem_query import validate_pubchem_compound, PUBCHEM_AVAILABLE


def _save_with_checksum(df: pd.DataFrame, filepath: str, metadata: dict) -> str:
    """保存DataFrame并生成校验和"""
    from src.io.integrity import save_checksum
    df.to_csv(filepath, index=False)
    save_checksum(filepath, metadata)
    return filepath


def step2_add_properties(
    input_csv: str,
    output_csv: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Step 2: 为有CID的化合物获取额外属性。
    
    Args:
        input_csv: 输入CSV路径（需包含cid列）
        output_csv: 输出CSV路径，默认为 input_csv_with_props.csv
        verbose: 是否打印进度
        
    Returns:
        str: 输出文件路径
    """
    if not PUBCHEM_AVAILABLE:
        raise ImportError("pubchempy not installed")
    
    input_path = Path(input_csv)
    if output_csv is None:
        stem = input_path.stem.replace("_with_cid", "")
        output_csv = str(input_path.with_name(stem + "_with_props.csv"))
    
    if verbose:
        print(f"\n[Step 2] Fetching additional properties...")
        print(f"  Input: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    if 'cid' not in df.columns:
        raise ValueError("Input CSV must contain 'cid' column. Run Step 1 first.")
    
    # Filter out rows with NaN or zero CID (invalid compounds)
    df_with_cid = df[(df['cid'].notna()) & (df['cid'] != 0)]
    
    if verbose:
        print(f"  Processing {len(df_with_cid)} compounds with valid CID...")
    
    for idx, row in tqdm(df_with_cid.iterrows(), total=len(df_with_cid), desc="Properties"):
        result = validate_pubchem_compound(
            row['SMILES'], verbose=False, get_properties=True
        )
        if result:
            for key, value in result.items():
                if key != 'cid':
                    df.at[idx, key] = value
    
    metadata = {
        "step": "properties_added",
        "total_compounds": len(df),
        "valid_cid_count": int(df['cid'].notna().sum()),
        "has_properties": True
    }
    _save_with_checksum(df, output_csv, metadata)
    
    if verbose:
        print(f"  [OK] Saved to: {output_csv}")
    
    return output_csv
