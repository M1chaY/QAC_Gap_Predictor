from pathlib import Path

import pandas as pd
from rdkit import Chem

from src import build_3d_mol


def generate_structure(
        smiles_list_path: str,
        output_path: str,
) -> None:
        """从包含 SMILES 列的 CSV 中批量生成 .mol 文件。

        参数:
                smiles_list_path: 含有 df['SMILES'] 列的 CSV 文件路径。
                output_path: 输出目录路径，每个 SMILES 生成一个 .mol 文件。
        """

        smiles_csv = Path(smiles_list_path)
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(smiles_csv)
        if "SMILES" not in df.columns:
                raise ValueError("输入 CSV 中缺少 'SMILES' 列")

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
                except Exception as e:  # rdkit 写文件可能抛异常
                        print(f"Failed to save mol for row {idx}: {e}")
                        failed += 1

        print(f"Saved {saved} mol files to {out_dir}")
        if failed:
                print(f"Failed for {failed} SMILES entries")
