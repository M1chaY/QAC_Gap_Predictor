"""
QM9预训练脚本 (使用统一的数据处理)

策略：
1. 检查本地是否有qm9_final.csv数据，如果没有提示用户执行extract_qm9_data.py
2. 使用R4NGapDataset统一加载QM9数据
3. 按70:15:15划分训练/验证/测试集
4. 使用验证集MAE作为early stopping指标进行预训练
5. 保存最佳预训练模型
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from src import R4NGapDataset, GapPredictionGNN, train_epoch, evaluate
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
QM9_PROCESSED_DIR = DATA_DIR / "qm9" / "processed_data"


def check_qm9_data():
    """
    检查qm9_final.csv是否存在
    """
    print("="*60)
    print("Step 1: Checking QM9 Final Data")
    print("="*60)
    
    qm9_final_path = QM9_PROCESSED_DIR / "qm9_final.csv"
    
    if qm9_final_path.exists():
        print(f"\n✓ QM9 final data found: {qm9_final_path}")
        # 显示数据信息
        df = pd.read_csv(qm9_final_path)
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")
        return True, qm9_final_path
    else:
        print(f"\n✗ QM9 final data not found: {qm9_final_path}")
        print("\nPlease run the following command first:")
        print("  python scripts/extract_qm9_data.py")
        print("\nThis will:")
        print("  1. Download QM9 dataset from PyG (if not present)")
        print("  2. Extract and process all QM9 data")
        print("  3. Filter molecules with F or valence errors")
        print("  4. Generate qm9_final.csv with SMILES and gap columns")
        return False, None


def pretrain_on_qm9(model, device, qm9_csv_path, num_epochs=100, batch_size=32):
    """
    在QM9数据集上预训练模型
    
    Args:
        model: GNN模型
        device: 训练设备
        qm9_csv_path: QM9数据CSV路径
        num_epochs: 训练轮数
        batch_size: 批次大小
    
    Returns:
        bool: 是否成功预训练
    """
    print("\n" + "="*60)
    print("Step 2: Pre-training on QM9 Dataset")
    print("="*60)
    
    try:
        # 使用R4NGapDataset加载QM9数据（统一的特征处理）
        print(f"\nLoading QM9 data from: {qm9_csv_path}")
        dataset = R4NGapDataset(qm9_csv_path)
        print(f"✓ Loaded {len(dataset)} QM9 samples")
        
        if len(dataset) == 0:
            print("✗ No valid QM9 samples!")
            return False
        
        # 数据划分: 70% 训练, 15% 验证, 15% 测试
        total_size = len(dataset)
        train_size = int(0.70 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nData split (70:15:15):")
        print(f"  Training:   {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
        print(f"  Validation: {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
        print(f"  Test:       {len(test_dataset)} ({len(test_dataset)/total_size*100:.1f}%)")
        
        # 优化器和学习率调度器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=5, verbose=True
        )
        
        # 预训练
        print("\n" + "="*60)
        print("Starting Pre-training...")
        print("="*60)
        print(f"Max epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Optimizer: Adam (lr=0.001, weight_decay=1e-5)")
        print(f"Early stopping: patience=10, monitor=val_mae")
        print("="*60)
        
        best_val_mae = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss = train_epoch(model, train_loader, optimizer, device)
            
            # 验证
            val_r2, val_mae, val_rmse, _, _ = evaluate(model, val_loader, device)
            
            # 学习率调度
            scheduler.step(val_mae)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 打印进度
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f} | "
                      f"Val R²={val_r2:.4f}, MAE={val_mae:.4f}, RMSE={val_rmse:.4f} | "
                      f"LR={current_lr:.6f}")
            
            # Early stopping (基于验证集MAE)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                # 保存最佳预训练模型
                best_model_path = MODEL_DIR / "qm9_pretrained.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                    'val_r2': val_r2,
                }, best_model_path)
                if epoch % 5 == 0 or epoch == 1:
                    print(f"  → Best model saved! (MAE improved: {val_mae:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n{'='*60}")
                    print(f"Early stopping at epoch {epoch}")
                    print(f"Best validation MAE: {best_val_mae:.4f}")
                    print(f"{'='*60}")
                    break
        
        # 加载最佳预训练模型
        print(f"\nLoading best model...")
        checkpoint = torch.load(MODEL_DIR / "qm9_pretrained.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 在测试集上评估
        print("\n" + "="*60)
        print("Final Evaluation on Test Set")
        print("="*60)
        test_r2, test_mae, test_rmse, _, _ = evaluate(model, test_loader, device)
        
        print(f"\nBest Validation Performance:")
        print(f"  MAE:  {checkpoint['val_mae']:.4f}")
        print(f"  R²:   {checkpoint['val_r2']:.4f}")
        print(f"  Epoch: {checkpoint['epoch']}")
        
        print(f"\nTest Set Performance:")
        print(f"  R²:   {test_r2:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        print(f"\n✓ Pre-training completed successfully!")
        print(f"✓ Best model saved to: {MODEL_DIR / 'qm9_pretrained.pt'}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pre-training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("QM9 Pre-training Script")
    print("="*60)
    
    # 创建模型目录
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Step 1: 检查QM9数据
    has_data, qm9_csv_path = check_qm9_data()
    
    if not has_data:
        print("\n" + "="*60)
        print("Pre-training aborted: QM9 data not found")
        print("="*60)
        sys.exit(1)
    
    # 读取一个样本以获取特征维度
    print("\nInitializing model...")
    temp_dataset = R4NGapDataset(qm9_csv_path)
    sample_data = temp_dataset[0]
    num_node_features = sample_data.x.shape[1]
    print(f"Node features: {num_node_features}")
    
    # 创建模型
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
    
    # Step 2: 在QM9上预训练
    success = pretrain_on_qm9(
        model=model,
        device=device,
        qm9_csv_path=qm9_csv_path,
        num_epochs=1000,
        batch_size=32
    )
    
    if success:
        print("\n" + "="*60)
        print("Pre-training Complete!")
        print("="*60)
        print(f"\nGenerated file:")
        print(f"  models/qm9_pretrained.pt - Pre-trained model checkpoint")
        print(f"\nNext steps:")
        print(f"  1. Use this pre-trained model for fine-tuning on R4N+ dataset")
        print(f"  2. Run: python scripts/finetune_gnn.py")
    else:
        print("\n" + "="*60)
        print("Pre-training Failed!")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()

