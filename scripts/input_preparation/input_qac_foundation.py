from src import preprocess_qac_dataset
from src.path import DATA_DIR, QAC_DIR


def main():
    preprocess_qac_dataset(
        smiles_csv=QAC_DIR / "dataset_qac_c20_with_cid_cleaned.csv",
        gap_xlsx=QAC_DIR / "qac_gap.xlsx",
        output_csv=DATA_DIR / "qac.csv",
        smiles_col="SMILES",
        target_col="gap"
    )


if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()
