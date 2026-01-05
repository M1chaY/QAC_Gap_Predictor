"""
QM9 数据集工作流脚本 (命令行版本)

合并了提取和特征生成功能的统一脚本。
支持三种模式：
1. 提取 QM9 数据集（从 PyG 下载并处理）
2. 预处理数据集（计算分子特征）
3. 完整工作流（提取 + 预处理）

如果需要在其他代码中调用，请使用:
    from src import extract_qm9, preprocess_qm9_dataset
"""

from src import extract_qm9, preprocess_qm9_dataset
from src.path import DATA_DIR, QM9_DIR


def extract_qm9_data():
    """提取 QM9 数据的流程"""
    print("=" * 50)
    print("QM9 Data Extraction")
    print("=" * 50)
    
    try:
        df = extract_qm9(QM9_DIR)
        print(f"\n[OK] QM9 extraction complete!")
        print(f"  Output: {QM9_DIR / 'qm9_final.csv'}")
        print(f"  Total samples: {len(df)}")
        return QM9_DIR / 'qm9_final.csv'
    except Exception as e:
        print(f"\n[ERROR] Error during extraction: {e}")
        return None


def preprocess_qm9_data():
    """预处理 QM9 数据（计算特征）的流程"""
    print("=" * 50)
    print("QM9 Data Preprocessing")
    print("=" * 50)
    
    # 检查输入文件
    input_file = QM9_DIR / "qm9_final.csv"
    if not input_file.exists():
        print(f"\n[ERROR] Input file not found: {input_file}")
        print("  Please run 'Extract QM9 data' first or ensure qm9_final.csv exists.")
        return None
    
    try:
        df = preprocess_qm9_dataset(
            input_csv=input_file,
            output_csv=DATA_DIR / "qm9.csv",
            smiles_col="SMILES",
            target_col="gap",
            skip_existing=False  # 总是重新计算
        )
        print(f"\n[OK] Preprocessing complete!")
        print(f"  Output: {DATA_DIR / 'qm9.csv'}")
        print(f"  Total samples: {len(df)}")
        return DATA_DIR / "qm9.csv"
    except Exception as e:
        print(f"\n[ERROR] Error during preprocessing: {e}")
        return None


def main():
    """主函数 - 提供菜单选择"""
    while True:
        print("\n" + "=" * 50)
        print("QM9 Dataset Workflow")
        print("=" * 50)
        print("Please select an option:")
        print("1. Extract QM9 data (from PyG)")
        print("2. Preprocess dataset (add features)")
        print("3. Full workflow (extract + preprocess)")
        print("0. Exit")
        print("=" * 50)
        
        choice = input("Your choice (0-3): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            extract_qm9_data()
        elif choice == '2':
            preprocess_qm9_data()
        elif choice == '3':
            print("\nStarting full workflow...")
            print("\n[Step 1/2] Extracting QM9 data...")
            result = extract_qm9_data()
            
            if result:
                print("\n[Step 2/2] Preprocessing dataset...")
                preprocess_qm9_data()
                print("\n[OK] Full workflow completed!")
            else:
                print("\n[ERROR] Workflow stopped due to extraction error.")
        else:
            print("Invalid choice. Please select 0-3.")
        
        # 询问是否继续
        continue_choice = input("\nDo you want to continue? (y/n): ")
        if continue_choice.lower() != 'y':
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()
