"""
PubChem CID验证

Step 1: 验证化合物在PubChem中是否存在，获取CID。
"""

import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.data.r4n.pubchem_query import validate_pubchem_compound, PUBCHEM_AVAILABLE

# 请求超时时间（秒）
REQUEST_TIMEOUT = 30
# 请求间隔（秒），避免过快请求被封
REQUEST_DELAY = 0.5


def _save_with_checksum(df: pd.DataFrame, filepath: str, metadata: dict) -> str:
    """保存DataFrame并生成校验和"""
    from src.io.integrity import save_checksum
    df.to_csv(filepath, index=False)
    save_checksum(filepath, metadata)
    return filepath


def step1_validate_cid(
    input_csv: str,
    output_csv: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Step 1: 验证PubChem CID。
    
    读取基础化合物CSV，验证CID，保存结果并生成校验和。
    
    Args:
        input_csv: 输入CSV路径（需包含SMILES列）
        output_csv: 输出CSV路径，默认为 input_csv_with_cid.csv
        verbose: 是否打印进度
        
    Returns:
        str: 输出文件路径
    """
    if not PUBCHEM_AVAILABLE:
        raise ImportError("pubchempy not installed")
    
    input_path = Path(input_csv)
    if output_csv is None:
        output_csv = str(input_path.with_name(input_path.stem + "_with_cid.csv"))
    
    if verbose:
        print(f"\n[Step 1] Validating PubChem CID...")
        print(f"  Input: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    results = []
    failed_indices = []
    skipped_smiles = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="CID Validation"):
        smiles = row['SMILES']
        try:
            # 每100条打印进度
            if verbose and idx % 100 == 0:
                print(f"\n  Processing {idx}/{len(df)}: {smiles[:30]}...")
            
            result = validate_pubchem_compound(
                smiles, verbose=False, get_properties=False
            )
            results.append(result if result else {})
            
            # 添加请求间隔，避免被PubChem限流
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            error_msg = str(e)
            # PubChem无法标准化结构的错误
            if "standardize" in error_msg.lower() or "structure" in error_msg.lower():
                if verbose:
                    print(f"\n  [SKIP] {idx}: {smiles[:30]}... - Cannot standardize")
                skipped_smiles.append((idx, smiles))
            else:
                if verbose:
                    print(f"\n  [WARN] {idx}: {smiles[:30]}... - {e}")
            results.append({})
            failed_indices.append(idx)
    
    if verbose:
        if skipped_smiles:
            print(f"\n  Skipped (unstandardizable): {len(skipped_smiles)}")
        if failed_indices:
            print(f"  Failed queries: {len(failed_indices)}")
    
    results_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    
    found = df['cid'].notna().sum()
    if verbose:
        print(f"\n  Found in PubChem: {found}/{len(df)}")
    
    metadata = {
        "step": "cid_validation",
        "total_compounds": len(df),
        "valid_cid_count": int(found)
    }
    _save_with_checksum(df, output_csv, metadata)
    
    if verbose:
        print(f"  [OK] Saved to: {output_csv}")
    
    return output_csv
