"""
R4N+微调脚本 (基于QM9预训练模型)

策略：
1. 检查本地是否有qm9_pretrained.pt模型，如果没有提示用户执行pretrain_gnn.py
2. 使用R4NGapDataset统一加载R4N+数据
3. 加载预训练模型并在R4N+数据集上微调
4. 使用K折交叉验证进行微调
5. 保存最佳微调模型和评估结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from src import R4NGapDataset, GapPredictionGNN, train_epoch, evaluate
from utils import metrics_to_df, scat_avp
from sklearn.model_selection import KFold
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def check_pretrained_model():
    """
    检查qm9_pretrained.pt是否存在
    """
    print("="*60)
    print("Step 1: Checking Pre-trained Model")
    print("="*60)
    
    pretrained_path = MODEL_DIR / "qm9_pretrained.pt"
    
    if pretrained_path.exists():
        print(f"\n✓ Pre-trained model found: {pretrained_path}")
        # 加载模型检查点以显示信息
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation MAE: {checkpoint.get('val_mae', 'N/A'):.4f}")
        print(f"  Validation R²: {checkpoint.get('val_r2', 'N/A'):.4f}")
        return True, pretrained_path
    else:
        print(f"\n✗ Pre-trained model not found: {pretrained_path}")
        print("\nPlease run the following command first:")
        print("  python scripts/pretrain_gnn.py")
        print("\nThis will:")
        print("  1. Load QM9 dataset from qm9_final.csv")
        print("  2. Pre-train GNN model on QM9 data")
        print("  3. Save the best pre-trained model as qm9_pretrained.pt")
        return False, None


def finetune_on_r4n(model, device, dataset, n_folds=5, num_epochs=200, batch_size=8):
    """
    在R4N+数据集上微调模型
    
    Args:
        model: 预训练的GNN模型
        device: 训练设备
        dataset: R4N+数据集
        n_folds: K折交叉验证的折数
        num_epochs: 最大训练轮数
        batch_size: 批次大小
    
    Returns:
        训练集和测试集的评估指标
    """
    print("\n" + "="*60)
    print("Step 2: Fine-tuning on R4N+ Dataset")
    print("="*60)
    
    # 划分测试集 (80% 训练, 20% 测试)
    total_size = len(dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nData split (80:20):")
    print(f"  Training: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
    print(f"  Test:     {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 微调时使用较小的学习率
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0015,  # 比预训练更小的学习率
        weight_decay=3e-5
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=25, verbose=True
    )
    
    # K折交叉验证微调
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    train_indices = list(range(len(train_dataset)))
    
    print(f"\n" + "="*60)
    print("Starting Fine-tuning...")
    print("="*60)
    print(f"Max epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"K-fold CV: {n_folds} folds")
    print(f"Optimizer: Adam (lr=0.0015, weight_decay=3e-5)")
    print(f"Early stopping: patience=40, monitor=val_mae")
    print("="*60)
    
    best_val_mae = float('inf')
    patience = 40
    patience_counter = 0
    
    for epoch in range(1, num_epochs + 1):
        fold_train_losses = []
        fold_val_metrics = {'r2': [], 'mae': [], 'rmse': []}
        
        # K折交叉验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices), 1):
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
            
            # 训练一个epoch
            train_loss = train_epoch(model, train_loader, optimizer, device)
            fold_train_losses.append(train_loss)
            
            # 验证
            val_r2, val_mae, val_rmse, _, _ = evaluate(model, val_loader, device)
            fold_val_metrics['r2'].append(val_r2)
            fold_val_metrics['mae'].append(val_mae)
            fold_val_metrics['rmse'].append(val_rmse)
        
        # 计算平均指标
        avg_train_loss = np.mean(fold_train_losses)
        avg_val_r2 = np.mean(fold_val_metrics['r2'])
        avg_val_mae = np.mean(fold_val_metrics['mae'])
        avg_val_rmse = np.mean(fold_val_metrics['rmse'])
        
        # 学习率调度
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{num_epochs}: "
                  f"Train Loss={avg_train_loss:.4f} | "
                  f"Val R²={avg_val_r2:.4f}, MAE={avg_val_mae:.4f}, RMSE={avg_val_rmse:.4f} | "
                  f"LR={current_lr:.6f}")
        
        # Early stopping (基于验证集MAE)
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            patience_counter = 0
            # 保存最佳微调模型
            best_model_path = MODEL_DIR / "r4n_finetuned.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': avg_val_mae,
                'val_r2': avg_val_r2,
            }, best_model_path)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  → Best model saved! (MAE improved: {avg_val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n{'='*60}")
                print(f"Early stopping at epoch {epoch}")
                print(f"Best validation MAE: {best_val_mae:.4f}")
                print(f"{'='*60}")
                break
    
    # 加载最佳微调模型
    print(f"\nLoading best fine-tuned model...")
    checkpoint = torch.load(MODEL_DIR / "r4n_finetuned.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终评估
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    full_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_r2, train_mae, train_rmse, train_pred, train_true = evaluate(model, full_train_loader, device)
    test_r2, test_mae, test_rmse, test_pred, test_true = evaluate(model, test_loader, device)
    
    print(f"\nBest Validation Performance:")
    print(f"  MAE:  {checkpoint['val_mae']:.4f}")
    print(f"  R²:   {checkpoint['val_r2']:.4f}")
    print(f"  Epoch: {checkpoint['epoch']}")
    
    print(f"\nTraining Set Performance:")
    print(f"  R²:   {train_r2:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    
    print(f"\nTest Set Performance:")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    
    print(f"\nOverfitting Gap: ΔR² = {train_r2 - test_r2:.4f}")
    
    return train_r2, train_mae, train_rmse, train_pred, train_true, test_r2, test_mae, test_rmse, test_pred, test_true


def main():
    print("R4N+ Fine-tuning Script")
    print("="*60)
    
    # 创建模型目录
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Step 1: 检查预训练模型
    has_pretrained, pretrained_path = check_pretrained_model()
    
    if not has_pretrained:
        print("\n" + "="*60)
        print("Fine-tuning aborted: Pre-trained model not found")
        print("="*60)
        sys.exit(1)
    
    # 加载R4N+数据集
    print("\nLoading R4N+ dataset...")
    csv_path = DATA_DIR / "r4n_dft.csv"
    
    if not csv_path.exists():
        print(f"\n✗ R4N+ dataset not found: {csv_path}")
        sys.exit(1)
    
    r4n_dataset = R4NGapDataset(csv_path)
    print(f"✓ Loaded {len(r4n_dataset)} R4N+ samples")
    
    # 获取特征维度
    sample_data = r4n_dataset[0]
    num_node_features = sample_data.x.shape[1]
    print(f"Node features: {num_node_features}")
    
    # 创建模型
    print("\nInitializing model...")
    model = GapPredictionGNN(
        num_node_features=num_node_features,
        hidden_channels=80,
        num_global_features=4,
        num_heads=4
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 加载预训练权重
    print(f"\nLoading pre-trained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Pre-trained weights loaded successfully")
    
    # Step 2: 在R4N+上微调
    train_r2, train_mae, train_rmse, train_pred, train_true, \
    test_r2, test_mae, test_rmse, test_pred, test_true = finetune_on_r4n(
        model=model,
        device=device,
        dataset=r4n_dataset,
        n_folds=5,
        num_epochs=200,
        batch_size=8
    )
    
    # 生成指标表
    print("\n" + "="*60)
    print("Generating Metrics Summary")
    print("="*60)
    
    metrics_df = metrics_to_df(
        train_true, train_pred,
        test_true, test_pred,
        model_name='GNN_Finetuned'
    )
    
    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))
    
    # 保存指标
    metrics_path = MODEL_DIR / "r4n_finetuned_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # 绘制预测图
    print("\nGenerating prediction plot...")
    plot_path = MODEL_DIR / "r4n_finetuned_prediction.png"
    scat_avp(
        train_true, train_pred, 
        test_true, test_pred,
        save_path=str(plot_path),
        axis_min=5.0,
        axis_max=11.0,
        model_name='GNN_Finetuned'
    )
    print(f"✓ Prediction plot saved to: {plot_path}")
    
    print("\n" + "="*60)
    print("Fine-tuning Complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. models/r4n_finetuned.pt - Fine-tuned model checkpoint")
    print(f"  2. models/r4n_finetuned_metrics.csv - Evaluation metrics")
    print(f"  3. models/r4n_finetuned_prediction.png - Prediction plot")
    print(f"\nModel is ready for inference!")


if __name__ == "__main__":
    main()
