"""
分子结构验证器

验证季铵离子分子结构的有效性。
"""

from rdkit import Chem
from rdkit.Chem import rdmolops


def validate_r4n_molecule(smiles: str) -> tuple:
    """
    验证季铵离子分子结构的有效性。
    
    检查条件：
    1. 分子可被RDKit解析
    2. 分子总电荷为+1
    3. 存在四价正氮原子
    
    Args:
        smiles: 分子的SMILES字符串
        
    Returns:
        tuple: (is_valid, mol) - 有效性标志和RDKit分子对象
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False, None

        if rdmolops.GetFormalCharge(mol) != 1:
            return False, None

        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == 'N' and
                    atom.GetFormalCharge() == 1 and
                    atom.GetDegree() == 4):
                return True, mol

        return False, None

    except Exception:
        return False, None


def get_canonical_smiles(mol: Chem.Mol) -> tuple:
    """
    获取分子的标准化SMILES和碳原子数。
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        tuple: (canonical_smiles, carbon_count)
    """
    canonical_smiles = Chem.MolToSmiles(mol)
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    return canonical_smiles, carbon_count
