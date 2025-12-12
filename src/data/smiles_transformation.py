from pathlib import Path
from typing import List, Optional
import joblib
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    GraphDescriptors,
    rdDistGeom,
    rdForceFieldHelpers,
    rdMolDescriptors,
)
from torch_geometric.data import Data, Dataset


def _extract_atom_features(atom: Chem.Atom, conf) -> list:
    """提取单个原子的特征。"""
    atom_num = atom.GetAtomicNum()
    atom_type_onehot = [
        1 if atom_num == 6 else 0,  # C
        1 if atom_num == 1 else 0,  # H
        1 if atom_num == 8 else 0,  # O
        1 if atom_num == 7 else 0,  # N
    ]

    pos = conf.GetAtomPosition(atom.GetIdx())
    coords = [pos.x, pos.y, pos.z]

    return atom_type_onehot + coords + [atom.GetFormalCharge(), atom.GetHybridization().real]


def _extract_bond_features(mol: Chem.Mol) -> tuple:
    """提取分子的键特征（边索引和边属性）。"""
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


def build_3d_mol(smiles: str) -> Optional[Chem.Mol]:
    """从SMILES生成3D分子结构。外部调用保留为公开接口。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 0xF00D
    embed_result = rdDistGeom.EmbedMolecule(mol, params)
    if embed_result == -1:
        embed_result = rdDistGeom.EmbedMolecule(mol)
        if embed_result == -1:
            return None

    try:
        mmff_result = rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
        if mmff_result == 1:
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
    except ValueError:
        try:
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
        except ValueError:
            pass

    return mol


def mol_to_graph(mol: Chem.Mol, gap: float, precomputed_features: Optional[list] = None) -> Optional[Data]:
    """
    将RDKit分子转换为PyG图数据
    
    节点特征: 原子类型(one-hot: C,H,O,N)、3D坐标、杂化类型、形式电荷
    边特征: 键类型
    全局特征: 分子量、LogP等描述符
    
    Args:
        mol: RDKit Mol对象（已加氢的3D构象分子）
        gap: HOMO-LUMO Gap目标值
        precomputed_features: 预计算的全局特征列表 [mol_weight, logp, num_rotatable_bonds, bertz_ct]
                             如果提供, 将直接使用而不重新计算
        
    Returns:
        PyG Data对象或None
    """
    if mol is None:
        return None
    
    # 获取全局特征
    if precomputed_features is not None:
        mol_descriptors = precomputed_features
    else:
        mol_no_h = Chem.RemoveHs(mol)
        mol_descriptors = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            GraphDescriptors.BertzCT(mol_no_h),
        ]
    
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
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, y=y)
    
    return data


_FEATURE_COLUMNS = ["mol_weight", "logp", "num_rotatable_bonds", "bertz_ct"]


def _ensure_valid_columns(df: pd.DataFrame, smiles_col: str, target_col: str) -> None:
    if smiles_col not in df.columns:
        raise ValueError(f"输入文件缺少'{smiles_col}'列")
    if target_col not in df.columns:
        raise ValueError(f"输入文件缺少'{target_col}'列")


def _clean_numeric_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df[df[target_col].notna()].copy()


def load_graph_dataset(
    csv_path: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap",
    use_precomputed: bool = True,
    use_cache: bool = False,
    cache_path: Optional[Path] = None,
) -> List[Data]:
    """
    从CSV文件加载分子图数据集
    
    Args:
        csv_path: CSV文件路径, 需包含SMILES和目标值列
        smiles_col: SMILES列名, 默认'SMILES'
        target_col: 目标值列名, 默认'gap'
        use_precomputed: 是否使用预计算的全局特征, 默认True
        
    Returns:
        PyG Data对象列表
    """

    csv_path = Path(csv_path)

    # 如果启用缓存并且缓存文件存在，则直接加载
    if use_cache:
        cache_path = Path(cache_path) if cache_path is not None else csv_path.with_suffix(".joblib")
        if cache_path.exists():
            print(f"从缓存加载图数据集: {cache_path}")
            return joblib.load(cache_path)

    df = pd.read_csv(csv_path)
    _ensure_valid_columns(df, smiles_col, target_col)
    df = _clean_numeric_target(df, target_col)

    print(f"加载了 {len(df)} 个有效样本")
    print(f"{target_col}范围: {df[target_col].min():.2f} - {df[target_col].max():.2f}")

    has_precomputed = all(col in df.columns for col in _FEATURE_COLUMNS)

    if use_precomputed and has_precomputed:
        print("使用预计算的全局特征")
    elif use_precomputed and not has_precomputed:
        print("警告: CSV文件缺少预计算特征, 将实时计算")
        print("提示: 先运行 preprocess_dataset() 函数预先计算特征可提高效率")
        has_precomputed = False
    else:
        print("实时计算全局特征")
        has_precomputed = False

    data_list: List[Data] = []
    failed = 0

    for _, row in df.iterrows():
        smiles = row[smiles_col]
        gap = float(row[target_col])

        mol = build_3d_mol(smiles)
        if mol is None:
            failed += 1
            continue

        precomputed_features = None
        if has_precomputed:
            precomputed_features = [float(row[col]) for col in _FEATURE_COLUMNS]

        graph_data = mol_to_graph(mol, gap, precomputed_features)
        if graph_data is not None:
            data_list.append(graph_data)

    print(f"成功创建了 {len(data_list)} 个图对象")
    if failed > 0:
        print(f"处理失败: {failed} 个分子")

    # 保存到joblib缓存，避免下次重复生成
    if use_cache:
        cache_path = Path(cache_path) if cache_path is not None else csv_path.with_suffix(".joblib")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"保存图数据集到缓存: {cache_path}")
        joblib.dump(data_list, cache_path)

    return data_list


class R4NGapDataset(Dataset):
    """基于CSV的图数据集封装，并支持joblib缓存。

    第一次会从CSV生成图数据并保存为.joblib；之后直接从joblib加载。
    """

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
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.use_precomputed = use_precomputed
        self.use_cache = use_cache
        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else self.csv_path.with_suffix(".joblib")
        )

        self._data_list: List[Data] = load_graph_dataset(
            csv_path=self.csv_path,
            smiles_col=self.smiles_col,
            target_col=self.target_col,
            use_precomputed=self.use_precomputed,
            use_cache=self.use_cache,
            cache_path=self.cache_path,
        )

    def len(self) -> int:  # type: ignore[override]
        return len(self._data_list)

    def get(self, idx: int) -> Data:  # type: ignore[override]
        return self._data_list[idx]