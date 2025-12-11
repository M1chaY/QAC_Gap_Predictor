"""
R4N+ 阳离子生成脚本 (命令行版本)

这是一个命令行交互脚本，用于生成季铵离子化合物。
如果需要在其他代码中调用，请使用: from utils import R4NGenerator
"""

from utils import R4NGenerator


def main():
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
    generator.save_results(
        compounds, 
        filename=f'data/dataset_r4n_c{max_carbons}.csv',
        validate_pubchem=validate_pubchem,
        get_properties=get_properties,
        get_halide_cas=get_halide_cas,
        verbose=True
    )

    print(f"\n{len(compounds)} R4N+ Cation Generated.")


if __name__ == "__main__":
    main()
    # 释放资源
    import gc
    gc.collect()