"""
分子结构文件生成器

从SMILES批量生成MOL文件。
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem

from src.molecule.builder import build_3d_mol


def generate_mol_files(smiles_csv_path: str, output_dir: str) -> None:
    """
    从包含SMILES列的CSV中批量生成MOL文件。

    Args:
        smiles_csv_path: 含有SMILES列的CSV文件路径
        output_dir: 输出目录路径，每个SMILES生成一个.mol文件
    """
    smiles_csv = Path(smiles_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(smiles_csv)
    if "SMILES" not in df.columns:
        raise ValueError("CSV missing 'SMILES' column")

    print(f"Loaded {len(df)} SMILES from {smiles_csv}")

    failed = 0
    saved = 0

    for idx, smiles in enumerate(df["SMILES"], start=1):
        if not isinstance(smiles, str) or not smiles.strip():
            failed += 1
            continue

        mol = build_3d_mol(smiles)
        if mol is None:
            failed += 1
            continue

        mol_filename = f"mol_{idx:03d}.mol"
        mol_path = out_dir / mol_filename

        try:
            Chem.MolToMolFile(mol, str(mol_path))
            saved += 1
        except Exception as e:
            print(f"Failed to save mol for row {idx}: {e}")
            failed += 1

    print(f"Saved {saved} mol files to {out_dir}")
    if failed:
        print(f"Failed for {failed} SMILES entries")
