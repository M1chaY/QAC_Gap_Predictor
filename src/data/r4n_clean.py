from pathlib import Path
from typing import Optional

import pandas as pd


def clean_r4n_data(csv_path: str) -> Optional[str]:
    """清洗 R4N 数据集并保存到新的 CSV。

    操作包括：
    - 删除 CID 为空的行；
    - 删除原始 "Index" 列（如果存在）；
    - 重置索引；

    结果会保存为原文件名追加 "_cleaned" 的 CSV, 并返回输出路径字符串。
    如果输入文件不存在则抛出 FileNotFoundError。
    """

    path = Path(csv_path)

    # 检查文件是否存在
    print(f"\nCleaning R4N data from {path}...")
    if not path.is_file():
        raise FileNotFoundError(f"The file {path} does not exist.")

    # 读取 R4N 数据集
    print("\nLoading R4N dataset for cleaning...")
    df = pd.read_csv(path)

    # 清洗所有 cid 为空的行
    print("Cleaning entries with missing cid...")
    before = len(df)
    df = df.dropna(subset=["cid"])  # 等价但更简洁
    after = len(df)
    print(f"Retained {after} entries with valid cid out of {before} total.")

    # 删除原来的 Index 列
    if "Index" in df.columns:
        print("Removing original Index column...")
        df = df.drop(columns=["Index"])

    # 重置索引
    df.reset_index(drop=True, inplace=True)

    # 保存清洗后的数据
    cleaned_csv_path = path.with_name(path.stem + "_with_cid.csv")
    print(f"\nSaving cleaned data to {cleaned_csv_path}...")
    df.to_csv(cleaned_csv_path, index=False)
    print("Cleaning complete.")
    print("=" * 50)

    return str(cleaned_csv_path)

def smiles_with_cid(csv_path: str) -> Optional[str]:
    """读取包含 CID 的 R4N 数据集并返回 DataFrame。

    如果输入文件不存在则抛出 FileNotFoundError。
    """
    path = Path(csv_path)

    # 检查文件是否存在
    print(f"Loading R4N dataset with CID from {path}...")
    if not path.is_file():
        raise FileNotFoundError(f"The file {path} does not exist.")

    # 读取 R4N 数据集
    df = pd.read_csv(path)
    df_smiles_cid = df[["SMILES", "cid"]]

    # 保存包含 smiles 和 cid 的数据
    smiles_cid_csv_path = path.with_name(path.stem + "_smiles_list.csv")
    print(f"\nSaving SMILES LIST to {smiles_cid_csv_path}...")
    df_smiles_cid.to_csv(smiles_cid_csv_path, index=False)
    print("SMILES LIST extraction complete.")
    print("=" * 50)


    return str(smiles_cid_csv_path)