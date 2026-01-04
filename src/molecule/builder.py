"""
3D分子构建器

从SMILES字符串构建3D分子结构。
"""

from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers


def build_3d_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    从SMILES生成3D分子结构。
    
    使用ETKDG算法生成初始构象，然后使用MMFF或UFF力场优化。
    
    Args:
        smiles: 分子的SMILES字符串
        
    Returns:
        Chem.Mol: 带有3D坐标的RDKit分子对象，失败返回None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # 使用ETKDG算法生成3D构象
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 0xF00D
    embed_result = rdDistGeom.EmbedMolecule(mol, params)
    
    if embed_result == -1:
        embed_result = rdDistGeom.EmbedMolecule(mol)
        if embed_result == -1:
            return None

    # 力场优化
    _optimize_geometry(mol)

    return mol


def _optimize_geometry(mol: Chem.Mol) -> None:
    """
    使用力场优化分子几何结构。
    
    优先使用MMFF力场，失败时使用UFF力场。
    
    Args:
        mol: RDKit分子对象
    """
    try:
        mmff_result = rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
        if mmff_result == 1:
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
    except ValueError:
        try:
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
        except ValueError:
            pass
