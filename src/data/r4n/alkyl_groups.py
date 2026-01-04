"""
烷基基团生成器

系统性生成烷基取代基。
"""

from typing import List, Tuple


def generate_alkyl_groups(max_carbons: int) -> List[Tuple[str, int]]:
    """
    系统性生成烷基基团。
    
    包括直链烷基和常见支链烷基。
    
    Args:
        max_carbons: 最大碳原子数
        
    Returns:
        list: [(smiles, carbon_count), ...] 烷基列表
    """
    alkyls = []

    # 直链烷基
    max_linear = max_carbons - 3
    for n in range(1, max_linear + 1):
        alkyls.append(("C" * n, n))
    
    # 支链烷基 C3-C8
    if max_carbons >= 3:
        branched_alkyls = _get_branched_alkyls()
        
        for n in range(3, min(max_carbons + 1, 9)):
            if n in branched_alkyls:
                for alkyl in branched_alkyls[n]:
                    alkyls.append((alkyl, n))

    return alkyls


def _get_branched_alkyls() -> dict:
    """
    获取支链烷基定义。
    
    Returns:
        dict: {carbon_count: [smiles_list]}
    """
    return {
        3: ["CC(C)"],
        4: ["CC(C)C"],
        5: ["CC(C)(C)C", "CCC(C)C", "CC(C)CC"],
        6: [
            "CC(C)(C)CC", "CC(C)(C)C(C)", 
            "CCCC(C)C", "CCC(C)CC", "CC(C)CCC"
        ],
        7: [
            "CC(C)(C)CCC", "CC(C)(C)C(C)C", 
            "CCCCC(C)C", "CCCC(C)CC", 
            "CCC(C)CCC", "CC(C)CCCC"
        ],
        8: [
            "CC(C)(C)CCCC", "CC(C)(C)C(C)CC", 
            "CCCCCC(C)C", "CCCCC(C)CC",
            "CCCC(C)CCC", "CCC(C)CCCC", "CC(C)CCCCC"
        ]
    }
