"""
R4N统计和保存工具

提供R4N化合物统计信息和文件保存功能。
"""

from typing import Dict, List

import pandas as pd


def get_statistics(compounds: List, carbon_distribution: Dict) -> Dict:
    """
    获取R4N化合物统计信息。
    
    Args:
        compounds: 化合物列表
        carbon_distribution: 碳原子分布字典
        
    Returns:
        dict: 统计信息
    """
    total = len(compounds)
    stats = {'total': total, 'distribution': {}}
    
    for carbons in sorted(carbon_distribution.keys()):
        count = carbon_distribution[carbons]
        percentage = (count / total * 100) if total > 0 else 0
        stats['distribution'][carbons] = {'count': count, 'percentage': percentage}
    
    return stats


def print_statistics(compounds: List, carbon_distribution: Dict) -> None:
    """
    打印R4N化合物统计信息。
    
    Args:
        compounds: 化合物列表
        carbon_distribution: 碳原子分布字典
    """
    stats = get_statistics(compounds, carbon_distribution)
    
    print(f"\nTotal R4N+ Cations: {stats['total']}")
    print(f"Carbon atom distribution:")
    
    for carbons, info in stats['distribution'].items():
        print(f"  {carbons}C: {info['count']} ({info['percentage']:.1f}%)")


def save_compounds_to_csv(
    compounds: List,
    filename: str,
    max_carbons: int
) -> str:
    """
    保存化合物到CSV文件。
    
    Args:
        compounds: 化合物列表，每个元素为(carbon_count, smiles)
        filename: 输出文件名，为None则自动生成
        max_carbons: 最大碳原子数
        
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
    df.to_csv(filename, index=False)
    
    print(f"\nResult saved to {filename}")
    
    return filename
