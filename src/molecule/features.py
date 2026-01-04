"""
分子特征计算器

计算分子的全局描述符特征。
"""

from typing import List

from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors


# 全局特征列名
FEATURE_COLUMNS = ["mol_weight", "num_rotatable_bonds", "bertz_ct"]


def compute_global_descriptors(mol: Chem.Mol) -> List[float]:
    """
    计算分子的全局描述符。
    
    Args:
        mol: RDKit分子对象（已加氢）
        
    Returns:
        list: [mol_weight, num_rotatable_bonds, bertz_ct]
    """
    mol_no_h = Chem.RemoveHs(mol)
    return [
        Descriptors.MolWt(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        GraphDescriptors.BertzCT(mol_no_h),
    ]


def compute_features_from_smiles(smiles: str) -> dict:
    """
    从SMILES计算全局特征。
    
    Args:
        smiles: 分子SMILES字符串
        
    Returns:
        dict: 特征字典，失败返回None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    mol_no_h = Chem.RemoveHs(mol)

    try:
        return {
            "mol_weight": Descriptors.MolWt(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "bertz_ct": GraphDescriptors.BertzCT(mol_no_h),
        }
    except Exception:
        return None
