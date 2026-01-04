"""
R4N数据清洗器

提供R4N数据集的清洗功能。
"""

from pathlib import Path
from typing import Optional

import pandas as pd


def clean_r4n_dataset(csv_path: str) -> Optional[str]:
    """
    清洗R4N数据集并保存到新的CSV。

    操作包括：
    - 删除CID为空的行
    - 删除原始Index列
    - 重置索引

    Args:
        csv_path: 输入CSV文件路径
        
    Returns:
        str: 输出文件路径
        
    Raises:
        FileNotFoundError: 文件不存在时
    """
    path = Path(csv_path)

    print(f"\nCleaning R4N data from {path}...")
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    print("\nLoading R4N dataset for cleaning...")
    df = pd.read_csv(path)

    print("Cleaning entries with missing cid...")
    before = len(df)
    df = df.dropna(subset=["cid"])
    after = len(df)
    print(f"Retained {after} entries with valid cid out of {before} total.")

    if "Index" in df.columns:
        print("Removing original Index column...")
        df = df.drop(columns=["Index"])

    df.reset_index(drop=True, inplace=True)

    cleaned_csv_path = path.with_name(path.stem + "_with_cid.csv")
    print(f"\nSaving cleaned data to {cleaned_csv_path}...")
    df.to_csv(cleaned_csv_path, index=False)
    print("Cleaning complete.")

    return str(cleaned_csv_path)
