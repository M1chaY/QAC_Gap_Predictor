"""
QM9 GNN预训练脚本

读取Optuna搜索得到的最佳网络配置进行最终训练。

策略：
1. 从JSON文件读取最佳网络结构和超参数
2. 数据集按70:15:15划分（与搜索阶段一致）
3. 最大迭代1000次，可变学习率
4. 100次耐心值早停机制
5. 保存最佳模型和训练曲线
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src import (
    PreparedGraphDataset, 
    GapPredictionGNN, 
    train_epoch, 
    evaluate, 
    plot_loss_curves,
    plot_actual_vs_predicted,
    set_seed,
    split_dataset
)

# 忽略torch-scatter警告
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# 固定随机种子（与搜索阶段一致）
RANDOM_SEED = 42


def load_config(config_path: Path) -> dict:
    """
    加载Optuna搜索结果配置。
    
    Args:
        config_path: JSON配置文件路径
        
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def pretrain_on_qm9(
    dataset_path: Path,
    config: dict,
    num_epochs: int = 1000,
    patience: int = 50,
    batch_size: int = 64
) -> bool:
    """
    使用最佳配置在QM9数据集上预训练。
    
    Args:
        dataset_path: 数据集路径
        config: 最佳超参数配置
        num_epochs: 最大训练轮数
        patience: 早停耐心值
        batch_size: 批次大小
        
    Returns:
        bool: 是否成功
    """
    try:
        # 设置随机种子确保可重复性
        set_seed(RANDOM_SEED)
        
        # 加载数据集
        print("\n" + "=" * 60)
        print("Loading dataset")
        print("=" * 60)
        
        dataset = PreparedGraphDataset(dataset_path)
        
        print(f"\nDataset info:")
        print(f"  Samples: {len(dataset)}")
        print(f"  Node features: {dataset.num_node_features}")
        print(f"  Edge features: {dataset.num_edge_features}")
        print(f"  Global features: {dataset.num_global_features}")
        
        # 划分数据集（与搜索阶段一致）
        train_data, val_data, test_data = split_dataset(
            dataset, train_ratio=0.7, val_ratio=0.15, seed=RANDOM_SEED
        )
        
        print(f"\nDataset split:")
        print(f"  Train: {len(train_data)} samples (70%)")
        print(f"  Val:   {len(val_data)} samples (15%)")
        print(f"  Test:  {len(test_data)} samples (15%)")
        
        # 创建DataLoader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # 设备配置
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {device}")
        
        # 从配置构建模型
        best_params = config['best_params']
        
        print("\n" + "=" * 60)
        print("Model configuration (from Optuna)")
        print("=" * 60)
        print(f"  GAT dims: {best_params['gat_dims']}")
        print(f"  MLP dims: {best_params['mlp_dims']}")
        print(f"  Attention heads: {best_params['num_heads']}")
        print(f"  Dropout: {best_params['dropout']}")
        print(f"  Learning rate: {best_params['lr']:.6f}")
        print(f"  Weight decay: {best_params['weight_decay']:.6f}")
        
        model = GapPredictionGNN(
            num_node_features=dataset.num_node_features,
            gat_dims=best_params['gat_dims'],
            mlp_dims=best_params['mlp_dims'],
            num_global_features=dataset.num_global_features,
            num_heads=best_params['num_heads'],
            dropout=best_params['dropout']
        ).to(device)
        
        total_params = model.count_parameters()
        print(f"\nModel parameters: {total_params:,}")
        
        # 优化器和学习率调度
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=best_params['lr'], 
            weight_decay=best_params['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
        )
        
        # 训练循环
        print("\n" + "=" * 60)
        print("Training")
        print("=" * 60)
        
        train_losses = []
        val_losses = []
        best_val_mae = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss = train_epoch(model, train_loader, optimizer, device)
            
            # 验证
            val_r2, val_mae, val_rmse, _, _ = evaluate(model, val_loader, device)
            
            # 记录损失
            train_losses.append(train_loss)
            val_losses.append(val_mae)
            
            # 学习率调度
            scheduler.step(val_mae)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 打印进度
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:04d}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val MAE={val_mae:.4f}, "
                      f"Val R2={val_r2:.4f} | "
                      f"LR={current_lr:.6f}")
            
            # 早停检查
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  -> Best model saved (Val MAE: {val_mae:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
                    break
        
        # 加载最佳模型
        model.load_state_dict(best_state)
        
        # 保存模型
        model_path = MODEL_DIR / "qm9_pretrained.pt"
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': best_params,
            'val_mae': best_val_mae,
            'dataset_info': {
                'num_node_features': dataset.num_node_features,
                'num_edge_features': dataset.num_edge_features,
                'num_global_features': dataset.num_global_features
            }
        }, model_path)
        print(f"\nModel saved: {model_path}")
        
        # 测试集评估
        print("\n" + "=" * 60)
        print("Test set evaluation")
        print("=" * 60)
        
        test_r2, test_mae, test_rmse, predictions, targets = evaluate(
            model, test_loader, device
        )
        
        print(f"\nTest set performance:")
        print(f"  R2:   {test_r2:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        # 过拟合分析
        overfitting_gap = test_mae - best_val_mae
        print(f"\nOverfitting analysis:")
        print(f"  MAE gap (test - val): {overfitting_gap:.4f}")
        if overfitting_gap < 0.05:
            print("  Status: Good generalization")
        elif overfitting_gap < 0.1:
            print("  Status: Acceptable")
        else:
            print("  Status: May need regularization")
        
        # 绘制损失曲线
        print("\n" + "=" * 60)
        print("Generating plots")
        print("=" * 60)
        
        loss_plot_path = MODEL_DIR / "qm9_pretrain_loss_curve.png"
        plot_loss_curves(
            train_losses=train_losses,
            test_losses=val_losses,
            save_path=loss_plot_path,
            title="QM9 Pretraining: Train vs Validation Loss"
        )
        print(f"Loss curve saved: {loss_plot_path}")
        
        # 绘制预测散点图
        scatter_plot_path = MODEL_DIR / "qm9_pretrain_scatter.png"
        plot_actual_vs_predicted(
            actual=targets,
            predicted=predictions,
            save_path=scatter_plot_path,
            title="QM9 Pretraining: Actual vs Predicted"
        )
        print(f"Scatter plot saved: {scatter_plot_path}")
        
        return True
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数。"""
    print("=" * 60)
    print("QM9 GNN Pretraining Script")
    print("=" * 60)
    
    # 创建输出目录
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    
    # 检查配置文件
    config_path = MODEL_DIR / "optuna_best_config.json"
    
    if not config_path.exists():
        print(f"\nConfig file not found: {config_path}")
        print("\nPlease run hyperparameter search first:")
        print("  python scripts/models/optuna_gnn_framework.py")
        sys.exit(1)
    
    print(f"\nConfig file found: {config_path}")
    
    # 加载配置
    config = load_config(config_path)
    
    # 检查数据集
    dataset_path = DATA_DIR / "qm9_prepared.joblib"
    
    if not dataset_path.exists():
        print(f"\nDataset not found: {dataset_path}")
        print("\nPlease run the following command first:")
        print("  python scripts/input_preparation/input_graph_preparation.py")
        sys.exit(1)
    
    print(f"Dataset found: {dataset_path}")
    
    # 开始训练
    success = pretrain_on_qm9(
        dataset_path=dataset_path,
        config=config,
        num_epochs=1000,
        patience=50,
        batch_size=64
    )
    
    if success:
        print("\n" + "=" * 60)
        print("Pretraining completed!")
        print("=" * 60)
        print(f"\nGenerated files:")
        print(f"  {MODEL_DIR / 'qm9_pretrained.pt'} - Pretrained model")
        print(f"  {MODEL_DIR / 'qm9_pretrain_loss_curve.png'} - Loss curve")
        print(f"  {MODEL_DIR / 'qm9_pretrain_scatter.png'} - Scatter plot")
    else:
        print("\n" + "=" * 60)
        print("Pretraining failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
