"""
Joblib数据集加载器

加载预处理好的joblib格式图数据集。
"""

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from torch_geometric.data import Data, Dataset


def load_prepared_dataset(dataset_path: Path) -> Tuple[List[Data], pd.DataFrame, dict]:
    """
    加载预处理好的图数据集。
    
    Args:
        dataset_path: joblib文件路径
        
    Returns:
        tuple: (graph_list, metadata, info)
            - graph_list: PyG Data对象列表
            - metadata: DataFrame
            - info: 数据集信息字典
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"Loading dataset: {dataset_path}")
    dataset = joblib.load(dataset_path)
    
    graph_list = dataset["graphs"]
    metadata = dataset["metadata"]
    
    info = {
        "smiles_col": dataset.get("smiles_col", "SMILES"),
        "target_col": dataset.get("target_col", "gap"),
        "num_samples": dataset.get("num_samples", len(graph_list))
    }
    
    print(f"Loaded: {info['num_samples']} graph objects")
    
    return graph_list, metadata, info


class PreparedGraphDataset(Dataset):
    """
    从joblib文件加载的图数据集封装。
    
    支持PyTorch Geometric的DataLoader。
    """
    
    def __init__(
        self,
        dataset_path: Path,
        transform=None,
        pre_transform=None
    ) -> None:
        super().__init__(None, transform, pre_transform)
        
        self.dataset_path = Path(dataset_path)
        self._data_list, self.metadata, self.info = load_prepared_dataset(
            self.dataset_path
        )
    
    def len(self) -> int:
        return len(self._data_list)
    
    def get(self, idx: int) -> Data:
        return self._data_list[idx]
    
    @property
    def num_node_features(self) -> int:
        """节点特征维度。"""
        return self._data_list[0].num_node_features
    
    @property
    def num_edge_features(self) -> int:
        """边特征维度。"""
        if self._data_list[0].edge_attr is not None:
            return self._data_list[0].num_edge_features
        return 0
    
    @property
    def num_global_features(self) -> int:
        """全局特征维度。"""
        if hasattr(self._data_list[0], 'u'):
            return self._data_list[0].u.shape[1]
        return 0
