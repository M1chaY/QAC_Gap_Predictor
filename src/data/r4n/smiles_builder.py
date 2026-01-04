"""
季铵离子SMILES构建器

构建R4N+季铵离子的SMILES字符串。
"""

from typing import List, Optional


def build_r4n_smiles(substituents: List[str]) -> Optional[str]:
    """
    构建季铵离子SMILES字符串。
    
    Args:
        substituents: 4个取代基的SMILES列表
        
    Returns:
        str: 季铵离子的SMILES字符串，无效时返回None
    """
    if len(substituents) != 4:
        return None

    main_chain = substituents[0]
    branches = substituents[1:]

    branch_part = ''.join(f'({r})' for r in branches)
    return f'[N+]{branch_part}{main_chain}'
