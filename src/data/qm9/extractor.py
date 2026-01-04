"""
QM9数据提取器

从PyG的QM9数据集中提取分子信息。
"""

import pandas as pd
from torch_geometric.datasets import QM9
from tqdm import tqdm


def extract_qm9_all_info(dataset: QM9) -> pd.DataFrame:
    """
    解析QM9数据集，获取所有信息。
    
    Args:
        dataset: PyG的QM9数据集对象
        
    Returns:
        pd.DataFrame: 包含所有QM9信息的DataFrame
    """
    print("\n" + "=" * 60)
    print("Step 3: Extracting All QM9 Information")
    print("=" * 60)

    print("\nExtracting all data fields...")
    data_list = []

    for idx, data in enumerate(tqdm(dataset, desc="Processing molecules")):
        smiles1 = data.smiles if hasattr(data, 'smiles') else None
        smiles2 = data.name if hasattr(data, 'name') else None
        
        targets = data.y[0].tolist() if hasattr(data, 'y') else [None] * 12
        
        atom_types = data.z.tolist() if hasattr(data, 'z') else []
        num_atoms = len(atom_types)
        unique_atoms = list(set(atom_types))
        
        data_list.append({
            'idx': data.idx.item() if hasattr(data, 'idx') else idx,
            'smiles': smiles1,
            'name': smiles2,
            'num_atoms': num_atoms,
            'atom_types': ','.join(map(str, atom_types)),
            'unique_atoms': ','.join(map(str, sorted(unique_atoms))),
            'mu': targets[0],
            'alpha': targets[1],
            'homo': targets[2],
            'lumo': targets[3],
            'gap': targets[4],
            'r2': targets[5],
            'zpve': targets[6],
            'U0': targets[7],
            'U': targets[8],
            'H': targets[9],
            'G': targets[10],
            'Cv': targets[11],
        })
    
    df = pd.DataFrame(data_list)
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")

    return df
