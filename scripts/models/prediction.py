"""
R4N Gap预测脚本

使用微调后的GNN模型对R4N数据集进行Gap预测。
读取Excel文件，为每个分子生成图数据并预测Gap值。
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import torch

from src import GapPredictionGNN, set_seed
from src.molecule.builder import build_3d_mol
from src.molecule.graph_converter import mol_to_graph

# 忽略torch-scatter警告
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# 固定随机种子
RANDOM_SEED = 42


def load_finetuned_model(model_path: Path, device: torch.device):
    """
    加载微调后的模型。
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
        
    Returns:
        tuple: (model, checkpoint)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_config = checkpoint['model_config']
    dataset_info = checkpoint['dataset_info']
    
    model = GapPredictionGNN(
        num_node_features=dataset_info['num_node_features'],
        gat_dims=model_config['gat_dims'],
        mlp_dims=model_config['mlp_dims'],
        num_global_features=dataset_info['num_global_features'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Test MAE: {checkpoint.get('test_mae', 'N/A')}")
    
    return model, checkpoint


def smiles_to_graph(smiles: str, global_features: list):
    """
    从SMILES生成图数据。
    
    Args:
        smiles: 分子SMILES字符串
        global_features: 预计算的全局特征 [mol_weight, num_rotatable_bonds, bertz_ct]
        
    Returns:
        Data: PyG图数据对象，失败返回None
    """
    mol = build_3d_mol(smiles)
    if mol is None:
        return None
    
    # 使用预计算的全局特征，gap设为0（预测时不使用）
    graph = mol_to_graph(mol, gap=0.0, precomputed_features=global_features)
    return graph


def predict_gaps(model, df: pd.DataFrame, device: torch.device) -> list:
    """
    对DataFrame中所有分子预测Gap值。
    
    Args:
        model: 加载好的GNN模型
        df: 包含SMILES和特征的DataFrame
        device: 计算设备
        
    Returns:
        list: 预测的Gap值列表
    """
    predictions = []
    
    print(f"\nPredicting {len(df)} molecules...")
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        
        # 提取全局特征
        global_features = [
            row['mol_weight'],
            row['num_rotatable_bonds'],
            row['bertz_ct']
        ]
        
        # 生成图数据
        graph = smiles_to_graph(smiles, global_features)
        
        if graph is None:
            print(f"  Warning: Failed to process row {idx + 1}: {smiles}")
            predictions.append(None)
            continue
        
        # 预测
        graph = graph.to(device)
        with torch.no_grad():
            pred = model(graph).item()
        
        predictions.append(pred)
        
        # 进度显示
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(df)} molecules")
    
    return predictions


def main():
    """主函数。"""
    print("=" * 60)
    print("R4N Gap Prediction Script")
    print("=" * 60)
    
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # 文件路径
    input_path = DATA_DIR / "r4n" / "r4n_gap.xlsx"
    model_path = MODEL_DIR / "r4n_finetuned.pt"
    output_path = MODEL_DIR / "r4n_c20_gap.csv"
    
    # 检查文件
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("Please run finetuning first:")
        print("  python scripts/models/gnn_finetune_onr4n.py")
        sys.exit(1)
    
    # 加载模型
    print("\n" + "=" * 60)
    print("Loading finetuned model")
    print("=" * 60)
    model, _ = load_finetuned_model(model_path, device)
    
    # 读取数据
    print("\n" + "=" * 60)
    print("Loading input data")
    print("=" * 60)
    df = pd.read_excel(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 预测
    print("\n" + "=" * 60)
    print("Running predictions")
    print("=" * 60)
    predictions = predict_gaps(model, df, device)
    
    # 添加预测结果列
    df['predicted_gap'] = predictions
    
    # 计算误差统计（仅对有真实值和预测值的行）
    valid_mask = df['predicted_gap'].notna() & df['gap'].notna()
    if valid_mask.sum() > 0:
        errors = abs(df.loc[valid_mask, 'gap'] - df.loc[valid_mask, 'predicted_gap'])
        print(f"\nPrediction statistics ({valid_mask.sum()} valid samples):")
        print(f"  MAE:  {errors.mean():.4f}")
        print(f"  Max Error: {errors.max():.4f}")
        print(f"  Min Error: {errors.min():.4f}")
    
    # 保存结果
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)
    df.to_csv(output_path, index=False)
    print(f"Results saved: {output_path}")
    
    # 显示前几行
    print("\nFirst 5 rows:")
    print(df[['Index', 'SMILES', 'gap', 'predicted_gap']].head().to_string())
    
    print("\n" + "=" * 60)
    print("Prediction completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
