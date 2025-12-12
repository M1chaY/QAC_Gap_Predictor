from src.path import R4N_DIR
from src import clean_r4n_data, smiles_with_cid

def main():
    r4n_csv_path = R4N_DIR / "dataset_r4n_c20.csv"
    cleaned_csv_path = clean_r4n_data(str(r4n_csv_path))
    print(f"Cleaned R4N data saved to: {cleaned_csv_path}")

    smiles_cid_csv_path = smiles_with_cid(str(cleaned_csv_path))
    print(f"SMILES LIST saved to: {smiles_cid_csv_path}")

if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()