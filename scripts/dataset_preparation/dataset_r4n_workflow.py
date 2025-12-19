"""
R4N+ 数据集工作流脚本 (命令行版本)

合并了生成和清洗功能的统一脚本。
支持三种模式：
1. 生成 R4N+ 化合物
2. 清洗 R4N+ 数据集
3. 生成并清洗（完整工作流）

如果需要在其他代码中调用，请使用:
    from src import R4NGenerator, clean_r4n_data
"""

from src import R4NGenerator, clean_r4n_data
from src.path import R4N_DIR


def generate_r4n_compounds():
    """生成 R4N+ 化合物的交互式流程"""
    print("=" * 50)
    print("R4N+ Cation Generator")
    print("LET'S GO, my friend!")
    print("=" * 50)

    # 获取用户输入
    while True:
        try:
            max_carbons = int(input("Plz input the max num of carbon atoms (Suggestion: 4-20): "))
            if max_carbons < 4:
                print("Error: R4N+ needs at least 4 carbons.")
                continue
            if max_carbons > 30:
                confirm = input("Warning: Too many max_carbons might take long time, continue？(y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("Error: plz input a valid integer.")

    # 生成化合物
    generator = R4NGenerator(max_carbons)
    compounds, carbon_distribution = generator.generate_compounds()

    # 显示统计信息
    generator.print_statistics(compounds, carbon_distribution)

    # 询问是否需要 PubChem 验证
    print("\n" + "=" * 50)
    validate = input("Do you want to validate with PubChem? (y/n): ")
    validate_pubchem = validate.lower() == 'y'
    
    get_properties = False
    get_halide_cas = False
    if validate_pubchem:
        props = input("Do you want to get additional properties? (y/n): ")
        get_properties = props.lower() == 'y'
        
        cas = input("Do you want to query halide salt CAS numbers? (y/n): ")
        get_halide_cas = cas.lower() == 'y'

    # 保存结果（可选 PubChem 验证）
    filename = f'data/r4n/dataset_r4n_c{max_carbons}.csv'
    generator.save_results(
        compounds, 
        filename=filename,
        validate_pubchem=validate_pubchem,
        get_properties=get_properties,
        get_halide_cas=get_halide_cas,
        verbose=True
    )

    print(f"\n{len(compounds)} R4N+ Cation Generated.")
    return filename


def clean_r4n_dataset():
    """清洗 R4N+ 数据集的交互式流程"""
    print("=" * 50)
    print("R4N+ Dataset Cleaner")
    print("=" * 50)
    
    # 获取要清洗的文件路径
    default_file = R4N_DIR / "dataset_r4n_c20.csv"
    file_input = input(f"Input CSV file path (default: {default_file}): ").strip()
    
    if not file_input:
        csv_path = default_file
    else:
        csv_path = file_input
    
    # 执行清洗
    try:
        cleaned_csv_path = clean_r4n_data(str(csv_path))
        print(f"\n✓ Cleaned R4N data saved to: {cleaned_csv_path}")
        return cleaned_csv_path
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return None


def main():
    """主函数 - 提供菜单选择"""
    while True:
        print("\n" + "=" * 50)
        print("R4N+ Dataset Workflow")
        print("=" * 50)
        print("Please select an option:")
        print("1. Generate R4N+ compounds")
        print("2. Clean R4N+ dataset")
        print("3. Generate and Clean (full workflow)")
        print("0. Exit")
        print("=" * 50)
        
        choice = input("Your choice (0-3): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            generate_r4n_compounds()
        elif choice == '2':
            clean_r4n_dataset()
        elif choice == '3':
            print("\n>>> Starting full workflow...")
            print("\n[Step 1/2] Generating R4N+ compounds...")
            generated_file = generate_r4n_compounds()
            
            print("\n[Step 2/2] Cleaning generated dataset...")
            # 询问是否清洗刚生成的文件
            clean_new = input(f"\nDo you want to clean the generated file? (y/n): ")
            if clean_new.lower() == 'y':
                clean_r4n_data(generated_file)
            else:
                clean_r4n_dataset()
            
            print("\n✓ Full workflow completed!")
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
