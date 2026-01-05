"""
输入数据处理工作流

该脚本用于处理用户输入的CSV或Excel文件，生成用于GNN训练的图特征数据集。

主要功能：
1. 读取CSV或Excel文件（支持指定Excel工作表）
2. 检查必需列（SMILES和gap）
3. 生成或使用预计算的全局分子特征（mol_weight, num_rotatable_bonds, bertz_ct）
4. 将SMILES转换为图特征
5. 保存为joblib格式供后续训练使用

使用方法：
    直接运行脚本，按提示输入相关信息：
    python input_graph_preparation.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import (
    load_input_file,
    validate_required_columns,
    clean_target_data,
    compute_global_features,
    convert_smiles_to_graphs,
    save_graph_dataset
)


def main():
    print("=" * 60)
    print("输入数据图特征准备工作流")
    print("=" * 60)
    print()
    
    try:
        # 1. 获取输入文件路径
        input_path = input("请输入数据文件路径（CSV或Excel）: ").strip()
        if not input_path:
            print("❌ 错误：文件路径不能为空")
            sys.exit(1)
        
        input_path = Path(input_path)
        
        # 2. 如果是Excel文件，获取工作表名
        sheet_name = None
        if input_path.suffix.lower() in [".xlsx", ".xls"]:
            sheet_name = input("请输入Excel工作表名: ").strip()
            if not sheet_name:
                print("❌ 错误：Excel文件必须指定工作表名")
                sys.exit(1)
        
        # 3. 获取输出文件路径
        default_output = input_path.parent / f"{input_path.stem}_prepared.joblib"
        output_input = input(f"请输入输出文件路径 (默认: {default_output}): ").strip()
        output_path = Path(output_input) if output_input else default_output
        
        # 4. 获取列名配置
        smiles_col = input("请输入SMILES列名 (默认: SMILES): ").strip() or "SMILES"
        target_col = input("请输入目标值列名 (默认: gap): ").strip() or "gap"
        
        print()
        print("配置信息:")
        print(f"  - 输入文件: {input_path}")
        if sheet_name:
            print(f"  - 工作表: {sheet_name}")
        print(f"  - 输出文件: {output_path}")
        print(f"  - SMILES列: {smiles_col}")
        print(f"  - 目标值列: {target_col}")
        print()
        
        confirm = input("确认开始处理？(y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消操作")
            sys.exit(0)
        
        print()
        print("=" * 60)
        
        # 5. 加载输入文件
        df = load_input_file(input_path, sheet_name)
        
        # 6. 验证必需列
        validate_required_columns(df, smiles_col, target_col)
        
        # 7. 清理目标值数据
        df = clean_target_data(df, target_col)
        
        # 8. 计算全局特征（如果需要）
        df = compute_global_features(df, smiles_col)
        
        # 9. 转换SMILES为图特征
        graph_list, df_valid = convert_smiles_to_graphs(
            df, smiles_col, target_col
        )
        
        if len(graph_list) == 0:
            print("❌ 错误：没有成功生成任何图对象")
            sys.exit(1)
        
        # 10. 保存数据集
        save_graph_dataset(
            graph_list, df_valid, output_path,
            smiles_col, target_col
        )
        
        print("=" * 60)
        print("✅ 工作流完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
