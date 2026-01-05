from src import preprocess_r4n_dataset
from src.path import DATA_DIR, R4N_DIR


def main():
    preprocess_r4n_dataset(
        smiles_csv=R4N_DIR / "dataset_r4n_c20_with_cid_cleaned.csv",
        gap_xlsx=R4N_DIR / "r4n_gap.xlsx",
        output_csv=DATA_DIR / "r4n.csv",
        smiles_col="SMILES",
        target_col="gap"
    )


if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()
