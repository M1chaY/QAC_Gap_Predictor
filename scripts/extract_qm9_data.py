"""
从QM9数据集中提取所有信息，保存为CSV格式
然后进行清洗和处理

处理流程:
1. 判断本地是否有qm9数据，如果有就跳过进入下一步；如果没有就从PyG的库中先下载源文件
2. 解析QM9数据集，获取所有信息，保存一份csv在本地（在qm9目录里新建一个目录保存）
3. 过滤带F元素的行
4. 处理smiles，转换为标准SMILES格式
5. 过滤掉有价态错误的分子（如Explicit valence错误）
6. 过滤掉带立体化学和手性的SMILES（包含@、/、\等符号）
7. 最终只保留SMILES和gap两列
"""

import pandas as pd
from pathlib import Path
from torch_geometric.datasets import QM9
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
QM9_DIR = DATA_DIR / "qm9"
QM9_PROCESSED_DIR = QM9_DIR / "processed_data"


def check_qm9_downloaded():
    """
    检查QM9数据是否已下载
    """
    print("="*60)
    print("Step 1: Checking QM9 Data Availability")
    print("="*60)
    
    # 检查PyG下载的原始文件
    qm9_raw_dir = QM9_DIR / "raw"
    qm9_processed_dir = QM9_DIR / "processed"
    
    has_data = False
    if qm9_raw_dir.exists() and qm9_processed_dir.exists():
        raw_files = list(qm9_raw_dir.glob("*"))
        processed_files = list(qm9_processed_dir.glob("*"))
        if len(raw_files) > 0 and len(processed_files) > 0:
            print(f"\n✓ QM9 data already downloaded")
            print(f"  Raw files: {len(raw_files)}")
            print(f"  Processed files: {len(processed_files)}")
            has_data = True
    
    if not has_data:
        print(f"\n✗ QM9 data not found, will download from PyG...")
    
    return has_data


def load_qm9_dataset():
    """
    加载或下载QM9数据集
    """
    print("\n" + "="*60)
    print("Step 2: Loading QM9 Dataset")
    print("="*60)
    
    print("\nLoading QM9 dataset (will download if not present)...")
    dataset = QM9(root=str(QM9_DIR))
    print(f"✓ Dataset loaded successfully")
    print(f"  Total molecules: {len(dataset)}")
    
    return dataset


def extract_qm9_all_info(dataset):
    """
    解析QM9数据集，获取所有信息
    """
    print("\n" + "="*60)
    print("Step 3: Extracting All QM9 Information")
    print("="*60)
    
    # 创建processed_data目录
    QM9_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\nExtracting all data fields...")
    data_list = []
    
    for idx in tqdm(range(len(dataset)), desc="Processing molecules"):
        data = dataset[idx]
        
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
    
    # 保存原始数据到qm9/processed_data目录
    raw_output = QM9_PROCESSED_DIR / "qm9_all_raw.csv"
    df.to_csv(raw_output, index=False)
    print(f"\n✓ Raw data saved to: {raw_output}")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {', '.join(df.columns)}")
    
    return df


def filter_fluorine(df):
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


def process_smiles_to_standard(df):
    """
    Step 4: 处理smiles，转换为标准SMILES格式
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
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES"):
        smiles = row['original_smiles']
        
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
                    'idx': row['idx'],
                    'SMILES': canonical_smiles,
                    'gap': row['gap'],
                    'num_atoms': row['num_atoms'],
                    'homo': row['homo'],
                    'lumo': row['lumo'],
                    'mu': row['mu'],
                    'alpha': row['alpha'],
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
    
    # 保存处理后的数据（包含更多列）
    cleaned_output = QM9_PROCESSED_DIR / "qm9_cleaned_full.csv"
    df_clean.to_csv(cleaned_output, index=False)
    print(f"\n✓ Cleaned data (full) saved to: {cleaned_output}")
    
    return df_clean


def filter_stereochemistry(df):
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
    
    # 保存过滤后的数据（包含更多列）
    no_stereo_output = QM9_PROCESSED_DIR / "qm9_no_stereochemistry.csv"
    df.to_csv(no_stereo_output, index=False)
    print(f"✓ Data without stereochemistry saved to: {no_stereo_output}")
    
    return df


def save_final_dataset(df):
    """
    Step 7: 只保留SMILES和gap两列
    """
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
    
    # 保存最终数据集
    final_output = QM9_PROCESSED_DIR / "qm9_final.csv"
    df_final.to_csv(final_output, index=False)
    print(f"\n✓ Final dataset saved to: {final_output}")
    
    # 显示前几个样本
    print(f"\nFirst 5 samples:")
    print(df_final.head())
    
    return df_final


if __name__ == "__main__":
    # Step 1: 检查QM9数据是否已下载
    has_data = check_qm9_downloaded()
    
    # Step 2: 加载或下载QM9数据集
    dataset = load_qm9_dataset()
    
    # Step 3: 提取所有信息并保存
    df_raw = extract_qm9_all_info(dataset)
    
    # Step 4: 过滤含F的分子
    df_no_f = filter_fluorine(df_raw)
    
    # Step 5: 处理SMILES并过滤价态错误
    df_clean = process_smiles_to_standard(df_no_f)
    
    # Step 6: 过滤立体化学和手性
    df_no_stereo = filter_stereochemistry(df_clean)
    
    # Step 7: 保存最终数据集（仅SMILES和gap）
    df_final = save_final_dataset(df_no_stereo)
    
    print("\n" + "="*60)
    print("QM9 Data Processing Complete!")
    print("="*60)
    print("\nGenerated files in data/qm9/processed_data/:")
    print("  1. qm9_all_raw.csv - All raw QM9 data with all properties")
    print("  2. qm9_cleaned_full.csv - Cleaned data (no F, no valence errors)")
    print("  3. qm9_no_stereochemistry.csv - Data without stereochemistry")
    print("  4. qm9_final.csv - Final dataset (SMILES and gap only)")
    print("\nYou can now use qm9_final.csv for your gap prediction tasks!")
