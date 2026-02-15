"""
季铵离子化合物生成器

系统性生成QAC季铵离子化合物。
"""

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import List, Tuple, Dict

from src.data.qac.smiles_builder import build_qac_smiles
from src.data.qac.alkyl_groups import generate_alkyl_groups
from src.data.qac.molecule_validator import validate_qac_molecule, get_canonical_smiles


class QACGenerator:
    """
    季铵离子(QAC)化合物生成器。
    
    系统性生成所有可能的季铵离子化合物组合。
    """

    def __init__(self, max_carbons: int):
        """
        初始化生成器。
        
        Args:
            max_carbons: 最大碳原子数限制
        """
        self.max_carbons = max_carbons
        self.alkyl_groups = generate_alkyl_groups(max_carbons)

    def generate_compounds(self, verbose: bool = True) -> Tuple[List, Dict]:
        """
        生成所有可能的季铵离子化合物。
        
        通过组合四个烷基取代基生成QAC化合物，并验证其有效性。
        
        Args:
            verbose: 是否打印进度信息
            
        Returns:
            tuple: (compounds, carbon_distribution)
                - compounds: 排序后的化合物列表，每个元素为(carbon_count, smiles)
                - carbon_distribution: 碳原子数分布字典
        """
        unique_compounds = set()
        carbon_distribution = defaultdict(int)

        if verbose:
            print(f"Using {len(self.alkyl_groups)} alkyl groups...")

        for combo in combinations_with_replacement(self.alkyl_groups, 4):
            total_carbons = sum(alkyl[1] for alkyl in combo)

            if total_carbons > self.max_carbons:
                continue

            substituents = [alkyl[0] for alkyl in combo]
            smiles = build_qac_smiles(substituents)
            
            if not smiles:
                continue

            is_valid, mol = validate_qac_molecule(smiles)
            if not is_valid:
                continue

            canonical_smiles, carbon_count = get_canonical_smiles(mol)

            if canonical_smiles not in {c[1] for c in unique_compounds}:
                unique_compounds.add((carbon_count, canonical_smiles))
                carbon_distribution[carbon_count] += 1

        return sorted(unique_compounds), dict(carbon_distribution)
