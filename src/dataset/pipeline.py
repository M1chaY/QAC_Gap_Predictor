"""
图转换流水线

将SMILES批量转换为图数据。
"""

from typing import List, Tuple

import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm

from src.molecule.builder import build_3d_mol
from src.molecule.graph_converter import mol_to_graph
from src.molecule.features import FEATURE_COLUMNS


def convert_smiles_to_graphs(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "gap"
) -> Tuple[List[Data], pd.DataFrame]:
    """
    将DataFrame中的SMILES批量转换为图特征。
    
    Args:
        df: 包含SMILES和特征的DataFrame
        smiles_col: SMILES列名
        target_col: 目标值列名
        
    Returns:
        tuple: (graph_list, df_valid)
    """
    print("Converting SMILES to graph features...")
    
    graph_list = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating graphs"):
        smiles = row[smiles_col]
        gap = float(row[target_col])
        
        mol = build_3d_mol(smiles)
        if mol is None:
            continue
        
        precomputed_features = [float(row[col]) for col in FEATURE_COLUMNS]
        
        graph_data = mol_to_graph(mol, gap, precomputed_features)
        if graph_data is None:
            continue
        
        graph_list.append(graph_data)
        valid_indices.append(idx)
    
    df_valid = df.loc[valid_indices].copy()
    
    failed = len(df) - len(graph_list)
    if failed > 0:
        print(f"{failed} molecules failed to convert")
    
    print(f"Successfully generated {len(graph_list)} graph objects")
    
    return graph_list, df_valid
