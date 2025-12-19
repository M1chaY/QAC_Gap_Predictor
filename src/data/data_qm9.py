import pandas as pd
from pathlib import Path
from torch_geometric.datasets import QM9
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    GraphDescriptors,
    rdMolDescriptors,
)
from typing import Optional


def _check_qm9(
        qm9_raw_dir: Path,
        qm9_processed_dir: Path
) -> bool:
    """检查 QM9 数据是否已下载。"""
    print("=" * 60)
    print("Step 1: Checking QM9 Data Availability")
    print("=" * 60)

    has_data = False
    if qm9_raw_dir.exists() and qm9_processed_dir.exists():
        raw_files = list(qm9_raw_dir.glob("*"))
        processed_files = list(qm9_processed_dir.glob("*"))
        if len(raw_files) > 0 and len(processed_files) > 0:
            print("\n✓ QM9 data already downloaded")
            print(f"  Raw files: {len(raw_files)}")
            print(f"  Processed files: {len(processed_files)}")
            has_data = True

    if not has_data:
        print("\n✗ QM9 data not found, will download from PyG...")

    return has_data


def _load_qm9(
        qm9_dir: Path
) -> QM9:
    """加载或下载 QM9 数据集。

    参数 ``qm9_dir`` 应该是包含 ``raw`` 和 ``processed`` 子目录的根目录，
    即传给 ``torch_geometric.datasets.QM9`` 的 ``root`` 路径。
    """
    print("\n" + "=" * 60)
    print("Step 2: Loading QM9 Dataset")
    print("=" * 60)

    print("\nLoading QM9 dataset (will download if not present)...")
    dataset = QM9(root=str(qm9_dir))
    print("✓ Dataset loaded successfully")
    print(f"  Total molecules: {len(dataset)}")

    return dataset


def _extract_qm9_all_info(
        dataset: QM9
        ) -> pd.DataFrame:
    """
    解析QM9数据集, 获取所有信息
    """
    print("\n" + "="*60)
    print("Step 3: Extracting All QM9 Information")
    print("="*60)

    print("\nExtracting all data fields...")
    data_list = []

    for idx, data in enumerate(tqdm(dataset, desc="Processing molecules")):
        # QM9有两个SMILES版本
        smiles1 = data.smiles if hasattr(data, 'smiles') else None
        smiles2 = data.name if hasattr(data, 'name') else None
        
        # 获取所有target值
        # Targets: [mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv]
        targets = data.y[0].tolist() if hasattr(data, 'y') else [None] * 12
        
        # 原子信息
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
            'mu': targets[0],           # Dipole moment
            'alpha': targets[1],        # Isotropic polarizability
            'homo': targets[2],         # HOMO energy
            'lumo': targets[3],         # LUMO energy
            'gap': targets[4],          # HOMO-LUMO gap
            'r2': targets[5],           # Electronic spatial extent
            'zpve': targets[6],         # Zero point vibrational energy
            'U0': targets[7],           # Internal energy at 0K
            'U': targets[8],            # Internal energy at 298K
            'H': targets[9],            # Enthalpy at 298K
            'G': targets[10],           # Free energy at 298K
            'Cv': targets[11],          # Heat capacity at 298K
        })
    
    df = pd.DataFrame(data_list)
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")

    return df


def _filter_fluorine(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Step 3: 过滤带F元素的行
    """
    print("\n" + "="*60)
    print("Step 4: Filtering Molecules with Fluorine")
    print("="*60)
    
    print(f"\nOriginal samples: {len(df)}")
    
    # F的原子序数是9
    def contains_fluorine(atom_str):
        if pd.isna(atom_str):
            return True  # 标记为包含F，后续会被过滤
        atoms = set(map(int, atom_str.split(',')))
        return 9 in atoms
    
    df['has_F'] = df['atom_types'].apply(contains_fluorine)
    f_count = df['has_F'].sum()
    
    print(f"  Molecules with F: {f_count}")
    print(f"  Molecules without F: {len(df) - f_count}")
    
    # 过滤掉包含F的分子
    df = df[~df['has_F']].copy()
    df = df.drop(columns=['has_F'])
    
    print(f"\n✓ After filtering: {len(df)} samples")
    
    return df


def _process_smiles_to_standard(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Step 4: 处理smiles, 转换为标准SMILES格式
    Step 5: 过滤掉有价态错误的分子
    """
    print("\n" + "="*60)
    print("Step 5: Converting to Standard SMILES")
    print("="*60)
    
    # 选择SMILES列（优先使用smiles列，如果为空则用name列）
    df['original_smiles'] = df['smiles'].fillna(df['name'])
    
    print(f"\nProcessing {len(df)} molecules...")
    print("  - Converting to standard SMILES format")
    print("  - Filtering molecules with valence errors")
    
    valid_data = []
    valence_errors = 0
    parse_errors = 0
    
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing SMILES"):
        smiles = row.original_smiles
        
        if pd.isna(smiles):
            parse_errors += 1
            continue
        
        try:
            # 尝试解析SMILES
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                parse_errors += 1
                continue
            
            # 尝试移除显式H并转换为标准SMILES
            # 这一步可能会触发价态错误
            mol = Chem.RemoveHs(mol)
            
            # 检查分子的价态
            try:
                # 尝试转换为SMILES，如果有价态错误会在这里抛出异常
                canonical_smiles = Chem.MolToSmiles(mol)
                
                # 再次验证生成的SMILES
                test_mol = Chem.MolFromSmiles(canonical_smiles)
                if test_mol is None:
                    valence_errors += 1
                    continue
                
                # 保存有效数据
                valid_data.append({
                    'idx': row.idx,
                    'SMILES': canonical_smiles,
                    'gap': row.gap,
                    'num_atoms': row.num_atoms,
                    'homo': row.homo,
                    'lumo': row.lumo,
                    'mu': row.mu,
                    'alpha': row.alpha,
                })
                
            except Exception as e:
                # 捕获价态错误
                error_msg = str(e)
                if 'valence' in error_msg.lower() or 'explicit' in error_msg.lower():
                    valence_errors += 1
                else:
                    parse_errors += 1
                continue
                
        except Exception as e:
            error_msg = str(e)
            if 'valence' in error_msg.lower() or 'explicit' in error_msg.lower():
                valence_errors += 1
            else:
                parse_errors += 1
            continue
    
    df_clean = pd.DataFrame(valid_data)
    
    print(f"\n✓ Processing complete:")
    print(f"  Valid molecules: {len(df_clean)}")
    print(f"  Valence errors: {valence_errors}")
    print(f"  Parse errors: {parse_errors}")
    print(f"  Total filtered: {len(df) - len(df_clean)}")
    
    return df_clean


def _filter_stereochemistry(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Step 6: 过滤掉带立体化学和手性的SMILES
    """
    print("\n" + "="*60)
    print("Step 6: Filtering Stereochemistry and Chirality")
    print("="*60)
    
    print(f"\nOriginal samples: {len(df)}")
    
    # 检测立体化学符号：@（手性）、/（顺式）、\（反式）
    def has_stereochemistry(smiles):
        if pd.isna(smiles):
            return True  # 标记为有立体化学，后续会被过滤
        # 检查是否包含立体化学符号
        stereo_symbols = ['@', '/', '\\']
        return any(symbol in smiles for symbol in stereo_symbols)
    
    df['has_stereo'] = df['SMILES'].apply(has_stereochemistry)
    stereo_count = df['has_stereo'].sum()
    
    print(f"  Molecules with stereochemistry: {stereo_count}")
    print(f"  Molecules without stereochemistry: {len(df) - stereo_count}")
    
    # 过滤掉包含立体化学的分子
    df = df[~df['has_stereo']].copy()
    df = df.drop(columns=['has_stereo'])
    
    print(f"\n✓ After filtering: {len(df)} samples")
    
    return df


def extract_qm9(
        qm9_dir: Path,
        ) -> pd.DataFrame:
    """从 PyG 的 QM9 数据生成精简 CSV。

    期望目录结构为::

        <QM9_ROOT>/
            raw/
            processed/

    其中 ``qm9_dir`` 即为 ``<QM9_ROOT>``，函数会自动在其下寻找
    ``raw`` 与 ``processed`` 目录，并将最终生成的 ``qm9_final.csv``
    直接保存在 ``<QM9_ROOT>`` 下。
    """

    qm9_raw_dir = qm9_dir / "raw"
    qm9_processed_dir = qm9_dir / "processed"

    # 先检查原始 QM9 数据是否就绪
    _check_qm9(qm9_raw_dir, qm9_processed_dir)

    # 最终输出文件路径（固定在 QM9 根目录下）
    final_output = qm9_dir / "qm9_final.csv"

    # 如果已经生成过 qm9_final.csv，则直接读取并返回，跳过后续所有步骤
    if final_output.exists():
        print("\n" + "=" * 60)
        print("Step 2: Existing Processed QM9 Dataset Detected")
        print("=" * 60)
        print(f"\n✓ Found existing qm9_final.csv, skip regeneration: {final_output}")
        df_final = pd.read_csv(final_output)
        print(f"  Loaded samples: {len(df_final)}")
        return df_final

    # 传给 PyG QM9 的根目录就是 qm9_dir
    qm9_root_dir = qm9_dir

    dataset = _load_qm9(qm9_root_dir)
    df = _extract_qm9_all_info(dataset)
    df = _filter_fluorine(df)
    df = _process_smiles_to_standard(df)
    df = _filter_stereochemistry(df)

    print("\n" + "="*60)
    print("Step 7: Creating Final Dataset")
    print("="*60)
    
    # 只保留SMILES和gap列
    df_final = df[['SMILES', 'gap']].copy()
    
    print(f"\nFinal dataset statistics:")
    print(f"  Total samples: {len(df_final)}")
    print(f"  Columns: {', '.join(df_final.columns)}")
    print(f"  Gap range: {df_final['gap'].min():.4f} - {df_final['gap'].max():.4f} eV")
    print(f"  Gap mean: {df_final['gap'].mean():.4f} eV")
    print(f"  Gap std: {df_final['gap'].std():.4f} eV")
    
    # 保存最终数据集：固定保存在 QM9 根目录下
    df_final.to_csv(final_output, index=False)
    print(f"\n✓ Final dataset saved to: {final_output}")
    
    # 显示前几个样本
    print(f"\nFirst 5 samples:")
    print(df_final.head())
    
    return df_final


# ==================== Feature Generation Functions ====================

_FEATURE_COLUMNS = ["mol_weight", "num_rotatable_bonds", "bertz_ct"]


def _ensure_valid_columns(df: pd.DataFrame, smiles_col: str, target_col: str) -> None:
    if smiles_col not in df.columns:
        raise ValueError(f"输入文件缺少'{smiles_col}'列")
    if target_col not in df.columns:
        raise ValueError(f"输入文件缺少'{target_col}'列")


def _clean_numeric_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df[df[target_col].notna()].copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    return df[df[target_col].notna()].copy()


def _compute_molecular_descriptors(smiles: str) -> Optional[dict]:
    """从SMILES提取分子级全局描述符。失败返回None。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    mol_no_h = Chem.RemoveHs(mol)

    try:
        return {
            "mol_weight": Descriptors.MolWt(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "bertz_ct": GraphDescriptors.BertzCT(mol_no_h),
        }
    except Exception as exc:  # pragma: no cover - RDKit errors are data-dependent
        print(f"Error computing descriptors for SMILES '{smiles}': {exc}")
        return None


def preprocess_dataset(
    input_csv: Path,
    output_csv: Path,
    smiles_col: str = "SMILES",
    target_col: str = "gap",
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    预处理数据集：计算全局特征并保存
    
    读取包含SMILES和目标值的CSV文件, 计算四个全局分子描述符, 
    将特征添加到数据集并保存为新的CSV文件。
    
    Args:
        input_csv: 输入CSV文件路径（需包含SMILES和目标值列）
        output_csv: 输出CSV文件路径（将包含全局特征）
        smiles_col: SMILES列名, 默认'SMILES'
        target_col: 目标值列名, 默认'gap'
        skip_existing: 如果输出文件已存在是否跳过, 默认True
        
    Returns:
        处理后的DataFrame
    """
    if skip_existing and output_csv.exists():
        print(f"输出文件已存在: {output_csv}")
        print("加载已有文件...")
        return pd.read_csv(output_csv)

    print(f"读取输入文件: {input_csv}")
    df = pd.read_csv(input_csv)
    _ensure_valid_columns(df, smiles_col, target_col)
    df = _clean_numeric_target(df, target_col)

    print(f"有效样本数: {len(df)}")
    print(f"{target_col}范围: {df[target_col].min():.4f} - {df[target_col].max():.4f}")

    print("计算分子全局特征...")
    features_list = []
    failed_indices = []

    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        descriptors = _compute_molecular_descriptors(smiles)

        if descriptors is None:
            failed_indices.append(idx)
            features_list.append({col: None for col in _FEATURE_COLUMNS})
        else:
            features_list.append(descriptors)

    features_df = pd.DataFrame(features_list)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    if failed_indices:
        print(f"警告: {len(failed_indices)}个分子特征计算失败, 已移除")
        df = df.dropna(subset=_FEATURE_COLUMNS)

    print(f"成功处理样本数: {len(df)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"结果已保存至: {output_csv}")

    return df
