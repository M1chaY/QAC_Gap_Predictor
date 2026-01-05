"""
R4N+ 数据集工作流脚本 (命令行版本)

支持两种主要模式：
1. 生成 R4N+ 化合物（基础生成，无 PubChem 验证）
2. 分步 PubChem 验证流水线:
   - Step 1: CID 验证 -> *_with_cid.csv
   - Clean: 清洗无效 CID -> *_cleaned.csv（可选，在 Step 1 后执行）
   - Step 2: 属性查询 -> *_with_props.csv
   - Step 3: 卤化盐 CAS 查询 -> *_with_cas.csv

如果需要在其他代码中调用，请使用:
    from src import R4NGenerator, clean_r4n_dataset
    from src import print_statistics, save_compounds_to_csv
    from src import (
        step1_validate_cid, step2_add_properties,
        step3_query_halide_cas
    )
"""

from src import (
    R4NGenerator,
    clean_r4n_dataset,
    print_statistics,
    save_compounds_to_csv,
    step1_validate_cid,
    step2_add_properties,
    step3_query_halide_cas,
)
from src.io.integrity import check_data_integrity
from src.path import R4N_DIR


def generate_r4n_compounds():
    """生成 R4N+ 化合物的交互式流程（仅基础生成，不含验证）"""
    print("=" * 50)
    print("R4N+ Cation Generator")
    print("=" * 50)

    # 获取用户输入
    while True:
        try:
            max_carbons = int(input("Input max carbon atoms (Suggestion: 4-20): "))
            if max_carbons < 4:
                print("Error: R4N+ needs at least 4 carbons.")
                continue
            if max_carbons > 30:
                confirm = input("Warning: Large max_carbons may take long. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("Error: Please input a valid integer.")

    # 检查是否已存在有效数据
    filename = str(R4N_DIR / f'dataset_r4n_c{max_carbons}.csv')
    if check_data_integrity(filename, verbose=True):
        regenerate = input("\nData already exists and is valid. Regenerate anyway? (y/n): ")
        if regenerate.lower() != 'y':
            print("Skipping regeneration.")
            return filename

    # 生成化合物
    generator = R4NGenerator(max_carbons)
    compounds, carbon_distribution = generator.generate_compounds()

    # 显示统计信息
    print_statistics(compounds, carbon_distribution)

    # 保存结果（基础CSV，不含PubChem验证）
    save_compounds_to_csv(compounds, filename, max_carbons)
    print(f"\n[OK] {len(compounds)} R4N+ cations saved to: {filename}")
    print("\nTip: Use option 2 to run PubChem validation pipeline.")
    return filename


def run_pubchem_validation():
    """分步 PubChem 验证的交互式流程

    三个独立步骤，每步保存带 MD5 校验的中间结果：
    1. CID 验证：验证化合物在 PubChem 中是否存在
    2. 属性查询：获取 MolecularWeight, IUPAC Name 等
    3. 卤化盐 CAS 查询：获取对应卤化物的 CAS 号
    """
    print("=" * 50)
    print("PubChem Validation Pipeline (Step-by-Step)")
    print("=" * 50)
    print("Steps:")
    print("  1. CID Validation -> *_with_cid.csv")
    print("  c. Clean invalid CIDs -> *_cleaned.csv (optional, after step 1)")
    print("  2. Add Properties -> *_with_props.csv")
    print("  3. Query Halide CAS -> *_with_cas.csv")
    print("=" * 50)

    # 先选择运行模式
    print("\nSelect mode:")
    print("  a. Run full pipeline (all steps with cleaning)")
    print("  1. Step 1 only: CID validation")
    print("  c. Clean only: Remove invalid CIDs (requires *_with_cid.csv)")
    print("  2. Step 2 only: Add properties (*_with_cid.csv or *_cleaned.csv)")
    print("  3. Step 3 only: Query halide CAS (requires *_with_props.csv)")

    mode = input("Your choice (a/1/c/2/3): ").strip().lower()
    if mode not in ('a', '1', 'c', '2', '3'):
        print("[ERROR] Invalid choice.")
        return None

    # 根据模式确定默认文件
    default_file = R4N_DIR / "dataset_r4n_c20.csv"
    if mode == 'c':
        default_file = R4N_DIR / "dataset_r4n_c20_with_cid.csv"
    elif mode == '2':
        # 优先 cleaned，其次 with_cid
        cleaned = R4N_DIR / "dataset_r4n_c20_cleaned.csv"
        default_file = cleaned if cleaned.exists() else R4N_DIR / "dataset_r4n_c20_with_cid.csv"
    elif mode == '3':
        default_file = R4N_DIR / "dataset_r4n_c20_with_props.csv"

    file_input = input(f"\nInput CSV file path (default: {default_file}): ").strip()
    input_csv = file_input if file_input else str(default_file)

    try:
        if mode == 'a':
            # 运行完整流程（含清洗）
            print("\nRunning full validation pipeline with cleaning...")
            # Step 1
            cid_csv = input_csv.replace('.csv', '_with_cid.csv')
            if check_data_integrity(cid_csv, verbose=True):
                print(f"  [SKIP] Step 1: {cid_csv} already valid.")
            else:
                step1_validate_cid(input_csv, cid_csv)
            # Clean
            cleaned_csv = cid_csv.replace('_with_cid.csv', '_cleaned.csv')
            if check_data_integrity(cleaned_csv, verbose=True):
                print(f"  [SKIP] Clean: {cleaned_csv} already valid.")
            else:
                cleaned_csv = clean_r4n_dataset(cid_csv)
            # Step 2
            props_csv = cleaned_csv.replace('_cleaned.csv', '_with_props.csv')
            if check_data_integrity(props_csv, verbose=True):
                print(f"  [SKIP] Step 2: {props_csv} already valid.")
            else:
                step2_add_properties(cleaned_csv, props_csv)
            # Step 3
            final_csv = props_csv.replace('_with_props.csv', '_with_cas.csv')
            if check_data_integrity(final_csv, verbose=True):
                print(f"  [SKIP] Step 3: {final_csv} already valid.")
            else:
                step3_query_halide_cas(props_csv, final_csv)
            print(f"\n[OK] Pipeline completed. Final output: {final_csv}")
        elif mode == '1':
            # 仅第一步
            output_csv = input_csv.replace('.csv', '_with_cid.csv')
            if check_data_integrity(output_csv, verbose=True):
                regen = input(f"\n{output_csv} already valid. Regenerate? (y/n): ").strip().lower()
                if regen != 'y':
                    print("[SKIP] Step 1 skipped.")
                else:
                    print(f"\nStep 1: Validating CIDs...")
                    step1_validate_cid(input_csv, output_csv)
                    print(f"[OK] Output saved to: {output_csv}")
            else:
                print(f"\nStep 1: Validating CIDs...")
                step1_validate_cid(input_csv, output_csv)
                print(f"[OK] Output saved to: {output_csv}")
            # 询问是否清洗
            do_clean = input("\nRun cleaner to remove invalid CIDs? (y/n): ").strip().lower()
            if do_clean == 'y':
                cleaned_csv = clean_r4n_dataset(output_csv)
                print(f"[OK] Cleaned output saved to: {cleaned_csv}")
        elif mode == 'c':
            # 仅清洗
            if '_with_cid' not in input_csv:
                cid_csv = input_csv.replace('.csv', '_with_cid.csv')
                print(f"Note: Using {cid_csv} as input (requires CID column).")
            else:
                cid_csv = input_csv
            cleaned_csv = cid_csv.replace('_with_cid.csv', '_cleaned.csv')
            if check_data_integrity(cleaned_csv, verbose=True):
                regen = input(f"\n{cleaned_csv} already valid. Regenerate? (y/n): ").strip().lower()
                if regen != 'y':
                    print("[SKIP] Clean skipped.")
                else:
                    cleaned_csv = clean_r4n_dataset(cid_csv)
                    print(f"[OK] Cleaned output saved to: {cleaned_csv}")
            else:
                cleaned_csv = clean_r4n_dataset(cid_csv)
                print(f"[OK] Cleaned output saved to: {cleaned_csv}")
        elif mode == '2':
            # 仅第二步（支持 *_with_cid.csv 或 *_cleaned.csv）
            if '_with_cid' in input_csv or '_cleaned' in input_csv:
                step2_input = input_csv
            else:
                # 优先查找 cleaned，否则用 with_cid
                cleaned_csv = input_csv.replace('.csv', '_cleaned.csv')
                cid_csv = input_csv.replace('.csv', '_with_cid.csv')
                from pathlib import Path
                if Path(cleaned_csv).exists():
                    step2_input = cleaned_csv
                    print(f"Note: Using {cleaned_csv} as input.")
                else:
                    step2_input = cid_csv
                    print(f"Note: Using {cid_csv} as input.")
            # 生成输出文件名
            output_csv = step2_input.replace('_cleaned.csv', '_with_props.csv')
            output_csv = output_csv.replace('_with_cid.csv', '_with_props.csv')
            if check_data_integrity(output_csv, verbose=True):
                regen = input(f"\n{output_csv} already valid. Regenerate? (y/n): ").strip().lower()
                if regen != 'y':
                    print("[SKIP] Step 2 skipped.")
                else:
                    print(f"\nStep 2: Adding properties...")
                    step2_add_properties(step2_input, output_csv)
                    print(f"[OK] Output saved to: {output_csv}")
            else:
                print(f"\nStep 2: Adding properties...")
                step2_add_properties(step2_input, output_csv)
                print(f"[OK] Output saved to: {output_csv}")
        elif mode == '3':
            # 仅第三步（支持 *_with_props.csv）
            if '_with_props' in input_csv:
                step3_input = input_csv
            else:
                # 尝试查找 with_props 文件
                props_csv = input_csv.replace('.csv', '_with_props.csv')
                props_csv = props_csv.replace('_with_cid.csv', '_with_props.csv')
                props_csv = props_csv.replace('_cleaned.csv', '_with_props.csv')
                step3_input = props_csv
                print(f"Note: Using {props_csv} as input (requires properties).")
            output_csv = step3_input.replace('_with_props.csv', '_with_cas.csv')
            if check_data_integrity(output_csv, verbose=True):
                regen = input(f"\n{output_csv} already valid. Regenerate? (y/n): ").strip().lower()
                if regen != 'y':
                    print("[SKIP] Step 3 skipped.")
                else:
                    print(f"\nStep 3: Querying halide CAS numbers...")
                    step3_query_halide_cas(step3_input, output_csv)
                    print(f"[OK] Output saved to: {output_csv}")
            else:
                print(f"\nStep 3: Querying halide CAS numbers...")
                step3_query_halide_cas(step3_input, output_csv)
                print(f"[OK] Output saved to: {output_csv}")
        else:
            print("[ERROR] Invalid choice.")
            return None
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        return None
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def main():
    """主函数 - 提供菜单选择"""
    while True:
        print("\n" + "=" * 50)
        print("R4N+ Dataset Workflow")
        print("=" * 50)
        print("Please select an option:")
        print("1. Generate R4N+ compounds (basic, no PubChem)")
        print("2. PubChem validation pipeline (with cleaning option)")
        print("0. Exit")
        print("=" * 50)
        
        choice = input("Your choice (0-2): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            generate_r4n_compounds()
        elif choice == '2':
            run_pubchem_validation()
        else:
            print("Invalid choice. Please select 0-2.")
        
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
