"""
模型结果可视化脚本

加载已训练的模型和训练历史，生成：
1. 损失曲线图
2. 预测散点图

支持两种模型的可视化：
1. QM9预训练模型
2. QAC微调模型

运行脚本后会提示选择要可视化的模型。
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src import (
    PreparedGraphDataset,
    GapPredictionGNN,
    evaluate,
    set_seed,
    split_dataset,
    plot_actual_vs_predicted,
    plot_loss_curves
)

# 忽略torch-scatter警告
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# 子目录路径
FIGURES_DIR = MODEL_DIR / "figures"
PARAMS_DIR = MODEL_DIR / "params"

# 固定随机种子（与训练阶段一致）
RANDOM_SEED = 42

# 模型配置映射
MODEL_CONFIGS = {
    "pretrain": {
        "model_path": MODEL_DIR / "qm9_pretrained.pt",
        "history_path": PARAMS_DIR / "qm9_pretrain_history.npz",
        "dataset_path": DATA_DIR / "qm9_prepared.joblib",
        "loss_plot_name": "qm9_pretrain_loss_curve.png",
        "scatter_plot_name": "qm9_pretrain_scatter.png",
        "title_prefix": "QM9 Pretraining",
        "train_script": "python scripts/models/gnn_pretrain_onqm9.py"
    },
    "finetune": {
        "model_path": MODEL_DIR / "qac_finetuned.pt",
        "history_path": PARAMS_DIR / "qac_finetune_history.npz",
        "dataset_path": DATA_DIR / "qac_prepared.joblib",
        "loss_plot_name": "qac_finetune_loss_curve.png",
        "scatter_plot_name": "qac_finetune_scatter.png",
        "title_prefix": "QAC Finetuning",
        "train_script": "python scripts/models/gnn_finetune_onqac.py"
    }
}


def plot_training_history(history_path: Path, config: dict) -> None:
    """
    绘制训练历史损失曲线。
    
    Args:
        history_path: 训练历史文件路径
        config: 模型配置字典
    """
    print("\n" + "=" * 60)
    print("Generating loss curves")
    print("=" * 60)
    
    # 加载训练历史
    history = np.load(history_path)
    
    # 根据模型类型获取对应的loss字段
    if "epoch_train_losses" in history.files:
        # QAC finetune格式
        train_losses = history['epoch_train_losses']
        val_losses = history['epoch_val_losses']
        best_epoch = int(history['best_epoch'])
        best_val_mae = float(history['cv_avg_mae'])
    else:
        # QM9 pretrain格式
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        best_epoch = int(history['best_epoch'])
        best_val_mae = float(history['best_val_mae'])
    
    print(f"\nTraining history:")
    print(f"  Total epochs: {len(train_losses)}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val MAE: {best_val_mae:.4f}")
    print(f"  Test R2: {history['test_r2']:.4f}")
    print(f"  Test MAE: {history['test_mae']:.4f}")
    print(f"  Test RMSE: {history['test_rmse']:.4f}")
    
    # 绘制损失曲线
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    loss_plot_path = FIGURES_DIR / config["loss_plot_name"]
    plot_loss_curves(
        train_losses=train_losses.tolist(),
        test_losses=val_losses.tolist(),
        save_path=loss_plot_path,
        title=f"{config['title_prefix']}: Train vs Validation Loss"
    )
    print(f"\nLoss curve saved: {loss_plot_path}")


def load_model_and_evaluate(
    model_path: Path,
    dataset_path: Path,
    config: dict,
    model_type: str,
    batch_size: int = 64
) -> None:
    """
    加载模型并生成评估图表。
    
    Args:
        model_path: 模型文件路径
        dataset_path: 数据集路径
        config: 模型配置字典
        model_type: 模型类型 ('pretrain' 或 'finetune')
        batch_size: 批次大小
    """
    # 设置随机种子
    set_seed(RANDOM_SEED)
    
    # 加载检查点
    print("\n" + "=" * 60)
    print("Loading model checkpoint")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model_config = checkpoint['model_config']
    
    # 获取best_epoch字段（两种格式兼容）
    best_epoch = checkpoint.get('epoch', checkpoint.get('best_epoch', 'N/A'))
    val_mae = checkpoint.get('val_mae', checkpoint.get('cv_avg_mae', 'N/A'))
    
    print(f"\nModel config:")
    print(f"  GAT dims: {model_config['gat_dims']}")
    print(f"  MLP dims: {model_config['mlp_dims']}")
    print(f"  Num heads: {model_config['num_heads']}")
    print(f"  Best epoch: {best_epoch}")
    if isinstance(val_mae, float):
        print(f"  Val MAE: {val_mae:.4f}")
    
    # 加载数据集
    print("\n" + "=" * 60)
    print("Loading dataset")
    print("=" * 60)
    
    dataset = PreparedGraphDataset(dataset_path)
    
    # 根据模型类型划分数据集
    is_finetune = (model_type == "finetune")
    
    if is_finetune:
        # QAC微调：使用与训练相同的划分方式
        np.random.seed(RANDOM_SEED)
        all_indices = np.random.permutation(len(dataset))
        num_samples = checkpoint.get('finetune_info', {}).get('num_samples', 100)
        train_indices = all_indices[:num_samples]
        test_indices = all_indices[num_samples:]
        train_data = [dataset[i] for i in train_indices]
        test_data = [dataset[i] for i in test_indices]
        val_data = []
    else:
        # QM9预训练：使用标准划分
        train_data, val_data, test_data = split_dataset(
            dataset, train_ratio=0.7, val_ratio=0.15, seed=RANDOM_SEED
        )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples")
    if val_data:
        print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    
    # 创建DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # 构建模型并加载权重
    model = GapPredictionGNN(
        num_node_features=dataset.num_node_features,
        gat_dims=model_config['gat_dims'],
        mlp_dims=model_config['mlp_dims'],
        num_global_features=dataset.num_global_features,
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel parameters: {model.count_parameters():,}")
    
    # 评估
    print("\n" + "=" * 60)
    print("Evaluating model")
    print("=" * 60)
    
    _, _, _, train_predictions, train_targets = evaluate(model, train_loader, device)
    test_r2, test_mae, test_rmse, test_predictions, test_targets = evaluate(
        model, test_loader, device
    )
    
    print(f"\nTest set performance:")
    print(f"  R2:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    
    # 绘制散点图
    print("\n" + "=" * 60)
    print("Generating scatter plot")
    print("=" * 60)
    
    # 计算坐标轴范围
    all_values = np.concatenate([train_targets, test_targets])
    axis_min = int(np.floor(all_values.min()))
    axis_max = int(np.ceil(all_values.max()))
    
    scatter_plot_path = FIGURES_DIR / config["scatter_plot_name"]
    plot_actual_vs_predicted(
        y_train=train_targets,
        y_train_pred=train_predictions,
        y_test=test_targets,
        y_test_pred=test_predictions,
        save_path=scatter_plot_path,
        axis_min=axis_min,
        axis_max=axis_max,
        title=f"{config['title_prefix']}: Actual vs Predicted"
    )
    print(f"Scatter plot saved: {scatter_plot_path}")


def select_model_type() -> str:
    """
    交互式选择要可视化的模型类型。
    
    Returns:
        str: 模型类型 ('pretrain' 或 'finetune')
    """
    print("\nSelect model to visualize:")
    print("  1. QM9 Pretrained Model")
    print("  2. QAC Finetuned Model")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == "1":
            return "pretrain"
        elif choice == "2":
            return "finetune"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def main():
    """主函数。"""
    print("=" * 60)
    print("Model Results Visualization Tool")
    print("=" * 60)
    
    # 交互式选择模型类型
    model_type = select_model_type()
    config = MODEL_CONFIGS[model_type]
    
    print("\n" + "=" * 60)
    print(f"{config['title_prefix']} Results Visualization")
    print("=" * 60)
    
    # 检查模型文件
    model_path = config["model_path"]
    history_path = config["history_path"]
    dataset_path = config["dataset_path"]
    
    if not model_path.exists():
        print(f"\nError: Model file not found: {model_path}")
        print(f"\nPlease run training first:")
        print(f"  {config['train_script']}")
        sys.exit(1)
    
    print(f"\nModel file found: {model_path}")
    
    # 检查数据集
    if not dataset_path.exists():
        print(f"\nError: Dataset not found: {dataset_path}")
        sys.exit(1)
    
    print(f"Dataset found: {dataset_path}")
    
    # 绘制训练历史（如果存在）
    if history_path.exists():
        print(f"History file found: {history_path}")
        plot_training_history(history_path, config)
    else:
        print(f"\nWarning: History file not found: {history_path}")
        print("Loss curves will not be generated.")
    
    # 加载模型并生成散点图
    load_model_and_evaluate(
        model_path=model_path,
        dataset_path=dataset_path,
        config=config,
        model_type=model_type,
        batch_size=64
    )
    
    print("\n" + "=" * 60)
    print("Visualization completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
