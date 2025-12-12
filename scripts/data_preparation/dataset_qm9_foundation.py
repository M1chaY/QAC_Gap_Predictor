from src import extract_qm9
from src.path import QM9_DIR

def main():
    extract_qm9(QM9_DIR)

if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()