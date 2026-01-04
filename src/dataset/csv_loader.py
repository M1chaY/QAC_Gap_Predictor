"""
CSV图数据集加载器

从CSV文件加载分子图数据集。
"""

from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from torch_geometric.data import Data, Dataset

from src.molecule.builder import build_3d_mol
from src.molecule.graph_converter import mol_to_graph
from src.molecule.features import FEATURE_COLUMNS


def load_graph_dataset(
    csv_path: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap",
    use_precomputed: bool = True,
    use_cache: bool = False,
    cache_path: Optional[Path] = None,
) -> List[Data]:
    """
    从CSV文件加载分子图数据集。
    
    Args:
        csv_path: CSV文件路径
        smiles_col: SMILES列名
        target_col: 目标值列名
        use_precomputed: 是否使用预计算的全局特征
        use_cache: 是否使用缓存
        cache_path: 缓存文件路径
        
    Returns:
        list: PyG Data对象列表
    """
    csv_path = Path(csv_path)

    if use_cache:
        cache_path = Path(cache_path) if cache_path else csv_path.with_suffix(".joblib")
        if cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            return joblib.load(cache_path)

    df = pd.read_csv(csv_path)
    
    if smiles_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Missing required columns")
    
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    print(f"Loaded {len(df)} valid samples")

    has_precomputed = all(col in df.columns for col in FEATURE_COLUMNS)
    use_features = use_precomputed and has_precomputed

    data_list: List[Data] = []
    failed = 0

    for _, row in df.iterrows():
        smiles = row[smiles_col]
        gap = float(row[target_col])

        mol = build_3d_mol(smiles)
        if mol is None:
            failed += 1
            continue

        precomputed = None
        if use_features:
            precomputed = [float(row[col]) for col in FEATURE_COLUMNS]

        graph_data = mol_to_graph(mol, gap, precomputed)
        if graph_data is not None:
            data_list.append(graph_data)

    print(f"Created {len(data_list)} graph objects")
    if failed > 0:
        print(f"Failed: {failed} molecules")

    if use_cache:
        cache_path = Path(cache_path) if cache_path else csv_path.with_suffix(".joblib")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data_list, cache_path)

    return data_list


class R4NGapDataset(Dataset):
    """基于CSV的图数据集封装，支持缓存。"""

    def __init__(
        self,
        csv_path: Path,
        smiles_col: str = "SMILES",
        target_col: str = "gap",
        use_precomputed: bool = True,
        use_cache: bool = True,
        cache_path: Optional[Path] = None,
        transform=None,
        pre_transform=None,
    ) -> None:
        super().__init__(None, transform, pre_transform)

        self.csv_path = Path(csv_path)
        self.cache_path = (
            Path(cache_path) if cache_path else self.csv_path.with_suffix(".joblib")
        )

        self._data_list = load_graph_dataset(
            csv_path=self.csv_path,
            smiles_col=smiles_col,
            target_col=target_col,
            use_precomputed=use_precomputed,
            use_cache=use_cache,
            cache_path=self.cache_path,
        )

    def len(self) -> int:
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        return self._data_list[idx]
