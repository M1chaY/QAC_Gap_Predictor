"""
R4N+ 分子结构生成脚本

从 SMILES 列表生成 3D 分子结构文件 (.mol)。
"""

from src import generate_mol_files
from src.path import R4N_DIR


def main():
    """主函数 - 从 SMILES CSV 生成 MOL 文件"""
    smiles_list_path = R4N_DIR / "dataset_r4n_c20_with_cid_with_cas.csv"
    output_path = R4N_DIR / "mol_files"

    print(f"Generating .mol files from {smiles_list_path} to {output_path}...")
    generate_mol_files(str(smiles_list_path), str(output_path))
    print("Generation complete.")

if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()