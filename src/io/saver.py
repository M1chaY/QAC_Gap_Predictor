"""
数据集保存器

提供图数据集保存功能。
"""

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from torch_geometric.data import Data


def save_graph_dataset(
    graph_list: List[Data],
    df: pd.DataFrame,
    output_path: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap"
) -> None:
    """
    保存图数据集为joblib格式。
    
    保存内容:
    - graphs: PyG Data对象列表
    - metadata: DataFrame
    - smiles_col: SMILES列名
    - target_col: 目标值列名
    - num_samples: 样本数量
    
    Args:
        graph_list: 图数据列表
        df: DataFrame元数据
        output_path: 输出文件路径
        smiles_col: SMILES列名
        target_col: 目标值列名
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset = {
        "graphs": graph_list,
        "metadata": df,
        "smiles_col": smiles_col,
        "target_col": target_col,
        "num_samples": len(graph_list)
    }
    
    print(f"Saving dataset to: {output_path}")
    joblib.dump(dataset, output_path, compress=3)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Save complete! File size: {file_size_mb:.2f} MB")
    print(f"  - Graph data: {len(graph_list)} objects")
    print(f"  - Metadata: {len(df)} rows x {len(df.columns)} columns")
