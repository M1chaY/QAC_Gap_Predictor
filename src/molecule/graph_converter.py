"""
图转换器

将RDKit分子对象转换为PyTorch Geometric图数据。
"""

from typing import List, Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.molecule.features import compute_global_descriptors


def mol_to_graph(
    mol: Chem.Mol, 
    gap: float, 
    precomputed_features: Optional[List[float]] = None
) -> Optional[Data]:
    """
    将RDKit分子转换为PyG图数据。
    
    节点特征: 原子类型(one-hot: C,H,O,N)、3D坐标、电荷、杂化类型
    边特征: 键类型
    全局特征: 分子量等描述符
    
    Args:
        mol: RDKit Mol对象（已加氢的3D构象分子）
        gap: HOMO-LUMO Gap目标值
        precomputed_features: 预计算的全局特征列表
        
    Returns:
        Data: PyG Data对象，失败返回None
    """
    if mol is None:
        return None
    
    if precomputed_features is not None:
        mol_descriptors = precomputed_features
    else:
        mol_descriptors = compute_global_descriptors(mol)
    
    conf = mol.GetConformer()
    
    # 提取原子特征
    atom_features = [_extract_atom_features(atom, conf) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 提取键特征
    edge_indices, edge_attrs = _extract_bond_features(mol)
    
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    u = torch.tensor([mol_descriptors], dtype=torch.float)
    y = torch.tensor([gap], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, y=y)


def _extract_atom_features(atom: Chem.Atom, conf) -> List:
    """
    提取单个原子的特征。
    
    Args:
        atom: RDKit原子对象
        conf: 分子构象对象
        
    Returns:
        list: 原子特征列表
    """
    atom_num = atom.GetAtomicNum()
    atom_type_onehot = [
        1 if atom_num == 6 else 0,  # C
        1 if atom_num == 1 else 0,  # H
        1 if atom_num == 8 else 0,  # O
        1 if atom_num == 7 else 0,  # N
    ]

    pos = conf.GetAtomPosition(atom.GetIdx())
    coords = [pos.x, pos.y, pos.z]

    return atom_type_onehot + coords + [
        atom.GetFormalCharge(), 
        atom.GetHybridization().real
    ]


def _extract_bond_features(mol: Chem.Mol) -> tuple:
    """
    提取分子的键特征。
    
    Args:
        mol: RDKit分子对象
        
    Returns:
        tuple: (edge_indices, edge_attrs)
    """
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices.append([i, j])
        edge_indices.append([j, i])

        bond_type = bond.GetBondTypeAsDouble()
        edge_attrs.append([bond_type])
        edge_attrs.append([bond_type])

    return edge_indices, edge_attrs
