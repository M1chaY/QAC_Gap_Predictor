"""
原子类型过滤器

过滤含有特定原子类型的分子。
"""

import pandas as pd


def filter_fluorine(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤含有氟(F)元素的分子。
    
    Args:
        df: 输入DataFrame，需包含atom_types列
        
    Returns:
        pd.DataFrame: 过滤后的DataFrame
    """
    print("\n" + "=" * 60)
    print("Step 4: Filtering Molecules with Fluorine")
    print("=" * 60)
    
    print(f"\nOriginal samples: {len(df)}")
    
    def contains_fluorine(atom_str):
        if pd.isna(atom_str):
            return True
        atoms = set(map(int, atom_str.split(',')))
        return 9 in atoms  # F的原子序数是9
    
    df['has_F'] = df['atom_types'].apply(contains_fluorine)
    f_count = df['has_F'].sum()
    
    print(f"  Molecules with F: {f_count}")
    print(f"  Molecules without F: {len(df) - f_count}")
    
    df = df[~df['has_F']].copy()
    df = df.drop(columns=['has_F'])
    
    print(f"\nAfter filtering: {len(df)} samples")
    
    return df


def filter_stereochemistry(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤含有立体化学和手性标记的分子。
    
    Args:
        df: 输入DataFrame，需包含SMILES列
        
    Returns:
        pd.DataFrame: 过滤后的DataFrame
    """
    print("\n" + "=" * 60)
    print("Step 6: Filtering Stereochemistry and Chirality")
    print("=" * 60)
    
    print(f"\nOriginal samples: {len(df)}")
    
    def has_stereochemistry(smiles):
        if pd.isna(smiles):
            return True
        stereo_symbols = ['@', '/', '\\']
        return any(symbol in smiles for symbol in stereo_symbols)
    
    df['has_stereo'] = df['SMILES'].apply(has_stereochemistry)
    stereo_count = df['has_stereo'].sum()
    
    print(f"  Molecules with stereochemistry: {stereo_count}")
    print(f"  Molecules without stereochemistry: {len(df) - stereo_count}")
    
    df = df[~df['has_stereo']].copy()
    df = df.drop(columns=['has_stereo'])
    
    print(f"\nAfter filtering: {len(df)} samples")
    
    return df
