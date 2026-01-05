"""
QM9数据处理流水线

整合QM9数据加载、提取、过滤功能的主入口。
"""

from pathlib import Path

import pandas as pd

from src.data.qm9.loader import check_qm9_data, load_qm9_dataset
from src.data.qm9.extractor import extract_qm9_all_info
from src.data.qm9.atom_filter import filter_fluorine, filter_stereochemistry
from src.data.qm9.smiles_converter import convert_to_standard_smiles


def extract_qm9(qm9_dir: Path) -> pd.DataFrame:
    """
    从PyG的QM9数据生成精简CSV。

    期望目录结构为::

        <QM9_ROOT>/
            raw/
            processed/

    Args:
        qm9_dir: QM9根目录路径
        
    Returns:
        pd.DataFrame: 包含SMILES和gap的最终数据集
    """
    from src.io.integrity import check_data_integrity, save_checksum
    
    qm9_raw_dir = qm9_dir / "raw"
    qm9_processed_dir = qm9_dir / "processed"

    check_qm9_data(qm9_raw_dir, qm9_processed_dir)

    final_output = qm9_dir / "qm9_final.csv"

    if final_output.exists():
        print("\n" + "=" * 60)
        print("Step 2: Checking Existing QM9 Dataset")
        print("=" * 60)
        
        # 验证数据完整性
        if check_data_integrity(str(final_output), verbose=True):
            print(f"\nLoading validated qm9_final.csv: {final_output}")
            df_final = pd.read_csv(final_output)
            print(f"  Loaded samples: {len(df_final)}")
            return df_final
        else:
            print("Existing data failed integrity check. Regenerating...")

    dataset = load_qm9_dataset(qm9_dir)
    df = extract_qm9_all_info(dataset)
    df = filter_fluorine(df)
    df = convert_to_standard_smiles(df)
    df = filter_stereochemistry(df)

    print("\n" + "=" * 60)
    print("Step 7: Creating Final Dataset")
    print("=" * 60)
    
    df_final = df[['SMILES', 'gap']].copy()
    
    print(f"\nFinal dataset statistics:")
    print(f"  Total samples: {len(df_final)}")
    print(f"  Gap range: {df_final['gap'].min():.4f} - {df_final['gap'].max():.4f} eV")
    print(f"  Gap mean: {df_final['gap'].mean():.4f} eV")
    
    df_final.to_csv(final_output, index=False)
    
    # 保存校验和
    metadata = {
        "type": "qm9_extracted",
        "total_samples": len(df_final),
        "gap_min": float(df_final['gap'].min()),
        "gap_max": float(df_final['gap'].max()),
        "gap_mean": float(df_final['gap'].mean())
    }
    save_checksum(str(final_output), metadata)
    
    print(f"\nFinal dataset saved to: {final_output}")
    
    return df_final
