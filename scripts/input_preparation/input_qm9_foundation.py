from src import preprocess_qm9_dataset
from src.path import DATA_DIR, QM9_DIR

def main():
    preprocess_qm9_dataset(
        input_csv=QM9_DIR / "qm9_final.csv",
        output_csv=DATA_DIR / "qm9.csv",
        smiles_col="SMILES",
        target_col="gap"
    )

if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()