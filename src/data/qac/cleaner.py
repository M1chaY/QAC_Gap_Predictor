"""
QAC数据清洗器

提供QAC数据集的清洗功能。
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.io.integrity import save_checksum


def clean_qac_dataset(csv_path: str) -> Optional[str]:
    """
    清洗QAC数据集并保存到新的CSV。

    操作包括：
    - 删除CID为0或NaN的行（无效化合物）
    - 重置索引

    Args:
        csv_path: 输入CSV文件路径（需包含cid列）
        
    Returns:
        str: 输出文件路径
        
    Raises:
        FileNotFoundError: 文件不存在时
    """
    path = Path(csv_path)

    print(f"\nCleaning QAC data from {path}...")
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    print("\nLoading QAC dataset for cleaning...")
    df = pd.read_csv(path)

    print("Cleaning entries with missing or zero cid...")
    before = len(df)
    # Remove rows with NaN or zero CID (invalid compounds)
    df = df[df["cid"].notna() & (df["cid"] != 0)]
    after = len(df)
    print(f"Retained {after} entries with valid cid out of {before} total.")

    df.reset_index(drop=True, inplace=True)

    cleaned_csv_path = path.with_name(path.stem + "_cleaned.csv")
    print(f"\nSaving cleaned data to {cleaned_csv_path}...")
    df.to_csv(cleaned_csv_path, index=False)

    # Save MD5 checksum
    metadata = {
        "step": "cleaned",
        "source_file": path.name,
        "original_count": before,
        "cleaned_count": after,
        "removed_count": before - after
    }
    save_checksum(str(cleaned_csv_path), metadata)
    print("Cleaning complete.")

    return str(cleaned_csv_path)
