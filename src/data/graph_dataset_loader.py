"""
图数据集加载工具

用于加载input_graph_preparation.py生成的joblib格式数据集
"""

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data, Dataset


def load_prepared_dataset(dataset_path: Path) -> Tuple[List[Data], pd.DataFrame, dict]:
    """
    加载预处理好的图数据集
    
    Args:
        dataset_path: joblib文件路径
        
    Returns:
        - graph_list: PyG Data对象列表
        - metadata: DataFrame（包含SMILES、gap和全局特征）
        - info: 数据集信息字典
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    print(f"加载数据集: {dataset_path}")
    dataset = joblib.load(dataset_path)
    
    graph_list = dataset["graphs"]
    metadata = dataset["metadata"]
    
    info = {
        "smiles_col": dataset.get("smiles_col", "SMILES"),
        "target_col": dataset.get("target_col", "gap"),
        "num_samples": dataset.get("num_samples", len(graph_list))
    }
    
    print(f"✓ 加载完成: {info['num_samples']} 个图对象")
    
    return graph_list, metadata, info


class PreparedGraphDataset(Dataset):
    """
    从joblib文件加载的图数据集封装
    
    支持PyTorch Geometric的DataLoader
    
    示例:
        dataset = PreparedGraphDataset("data/prepared.joblib")
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for batch in loader:
            batch = batch.to(device)
            # 训练代码
    """
    
    def __init__(
        self,
        dataset_path: Path,
        transform=None,
        pre_transform=None
    ) -> None:
        super().__init__(None, transform, pre_transform)
        
        self.dataset_path = Path(dataset_path)
        self._data_list, self.metadata, self.info = load_prepared_dataset(self.dataset_path)
    
    def len(self) -> int:  # type: ignore[override]
        return len(self._data_list)
    
    def get(self, idx: int) -> Data:  # type: ignore[override]
        data = self._data_list[idx]
        return data
    
    @property
    def num_node_features(self) -> int:
        """节点特征维度"""
        return self._data_list[0].num_node_features
    
    @property
    def num_edge_features(self) -> int:
        """边特征维度"""
        return self._data_list[0].num_edge_features if self._data_list[0].edge_attr is not None else 0
    
    @property
    def num_global_features(self) -> int:
        """全局特征维度"""
        return self._data_list[0].u.shape[1] if hasattr(self._data_list[0], 'u') else 0
