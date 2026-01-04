"""
SMILES标准化转换器

将SMILES转换为标准格式并过滤无效分子。
"""

import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def convert_to_standard_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    将SMILES转换为标准格式并过滤价态错误的分子。
    
    Args:
        df: 输入DataFrame，需包含smiles或name列
        
    Returns:
        pd.DataFrame: 处理后的DataFrame，包含标准SMILES
    """
    print("\n" + "=" * 60)
    print("Step 5: Converting to Standard SMILES")
    print("=" * 60)
    
    df['original_smiles'] = df['smiles'].fillna(df['name'])
    
    print(f"\nProcessing {len(df)} molecules...")
    print("  - Converting to standard SMILES format")
    print("  - Filtering molecules with valence errors")
    
    valid_data = []
    valence_errors = 0
    parse_errors = 0
    
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing"):
        smiles = row.original_smiles
        
        if pd.isna(smiles):
            parse_errors += 1
            continue
        
        result = _process_single_smiles(smiles, row)
        if result is None:
            parse_errors += 1
        elif result == "valence_error":
            valence_errors += 1
        else:
            valid_data.append(result)
    
    df_clean = pd.DataFrame(valid_data)
    
    print(f"\nProcessing complete:")
    print(f"  Valid molecules: {len(df_clean)}")
    print(f"  Valence errors: {valence_errors}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Total filtered: {len(df) - len(df_clean)}")
    
    return df_clean


def _process_single_smiles(smiles: str, row) -> dict:
    """
    处理单个SMILES字符串。
    
    Args:
        smiles: SMILES字符串
        row: 数据行
        
    Returns:
        dict或None或"valence_error"
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        mol = Chem.RemoveHs(mol)
        
        try:
            canonical_smiles = Chem.MolToSmiles(mol)
            test_mol = Chem.MolFromSmiles(canonical_smiles)
            if test_mol is None:
                return "valence_error"
            
            return {
                'idx': row.idx,
                'SMILES': canonical_smiles,
                'gap': row.gap,
                'num_atoms': row.num_atoms,
                'homo': row.homo,
                'lumo': row.lumo,
                'mu': row.mu,
                'alpha': row.alpha,
            }
            
        except Exception as e:
            if 'valence' in str(e).lower() or 'explicit' in str(e).lower():
                return "valence_error"
            return None
            
    except Exception as e:
        if 'valence' in str(e).lower() or 'explicit' in str(e).lower():
            return "valence_error"
        return None
