"""
QM9数据集加载器

提供QM9数据集的检查和加载功能。
"""

from pathlib import Path

from torch_geometric.datasets import QM9


def check_qm9_data(qm9_raw_dir: Path, qm9_processed_dir: Path) -> bool:
    """
    检查QM9数据是否已下载。
    
    Args:
        qm9_raw_dir: QM9原始数据目录
        qm9_processed_dir: QM9处理后数据目录
        
    Returns:
        bool: 数据是否已存在
    """
    print("=" * 60)
    print("Step 1: Checking QM9 Data Availability")
    print("=" * 60)

    has_data = False
    if qm9_raw_dir.exists() and qm9_processed_dir.exists():
        raw_files = list(qm9_raw_dir.glob("*"))
        processed_files = list(qm9_processed_dir.glob("*"))
        if len(raw_files) > 0 and len(processed_files) > 0:
            print("\nQM9 data already downloaded")
            print(f"  Raw files: {len(raw_files)}")
            print(f"  Processed files: {len(processed_files)}")
            has_data = True

    if not has_data:
        print("\nQM9 data not found, will download from PyG...")

    return has_data


def load_qm9_dataset(qm9_dir: Path) -> QM9:
    """
    加载或下载QM9数据集。

    Args:
        qm9_dir: QM9根目录，包含raw和processed子目录
        
    Returns:
        QM9: PyG的QM9数据集对象
    """
    print("\n" + "=" * 60)
    print("Step 2: Loading QM9 Dataset")
    print("=" * 60)

    print("\nLoading QM9 dataset (will download if not present)...")
    dataset = QM9(root=str(qm9_dir))
    print("Dataset loaded successfully")
    print(f"  Total molecules: {len(dataset)}")

    return dataset
