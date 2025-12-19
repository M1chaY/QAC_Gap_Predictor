from src import generate_structure
from src.path import R4N_DIR
from rdkit import Chem

def main():
    smiles_list_path = R4N_DIR / "dataset_r4n_c20_with_cid_smiles_list.csv"
    output_path = R4N_DIR / "mol_files"

    print(f"Generating .mol files from {smiles_list_path} to {output_path}...")
    generate_structure(str(smiles_list_path), str(output_path))
    print("Generation complete.")

if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()