"""
QM9 预训练脚本

使用预处理好的 joblib 数据集进行 GNN 模型预训练

策略：
1. 加载预处理的 QM9 数据集（joblib 格式）
2. 按 70:30 划分训练/测试集
3. 训练集使用 5-Fold 交叉验证，以平均 MAE 作为损失指标
4. 保存最佳预训练模型到 models/qm9_pretrained.pt
"""

import sys
from pathlib import Path
import warnings
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

from src import PreparedGraphDataset, GapPredictionGNN, train_epoch, evaluate, compute_loss, plot_loss_curves

# 忽略torch-scatter警告（性能优化提示，不影响功能）
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


def pretrain_on_qm9(dataset_path, num_epochs=1000, batch_size=32, n_folds=3):
    """
    在 QM9 数据集上预训练模型（使用 5-Fold 交叉验证）

    Args:
        dataset_path: QM9 预处理数据集路径（joblib 文件）
        num_epochs: 最大训练轮数
        batch_size: 批次大小
        n_folds: 交叉验证折数

    Returns:
        bool: 是否成功预训练
    """
    try:
        # 加载数据集
        print("\n" + "=" * 60)
        print("加载预处理数据集")
        print("=" * 60)

        dataset = PreparedGraphDataset(dataset_path)

        print(f"\n数据集信息:")
        print(f"  - 样本数量: {len(dataset)}")
        print(f"  - 节点特征维度: {dataset.num_node_features}")
        print(f"  - 边特征维度: {dataset.num_edge_features}")
        print(f"  - 全局特征维度: {dataset.num_global_features}")

        # 划分数据集 (70:30)
        train_size = int(0.7 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        print(f"\n数据集划分:")
        print(f"  - 训练集: {len(train_dataset)} 样本 (70%)")
        print(f"  - 测试集: {len(test_dataset)} 样本 (30%)")
        print(f"  - 交叉验证: {n_folds}-Fold")

        # 创建测试集 DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"\nDataLoader 配置:")
        print(f"  - Batch Size: {batch_size}")

        # 初始化设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n使用设备: {device}")

        # K-Fold 设置
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        train_indices = list(range(len(train_dataset)))
        
        best_cv_mae = float('inf')
        patience_counter = 0
        patience = 15
        best_epoch = 0
        
        # 初始化一个模型用于最终训练（基于最佳超参数）
        final_model = GapPredictionGNN(
            num_node_features=dataset.num_node_features,
            hidden_channels=64,  # 减小到64
            num_global_features=dataset.num_global_features,
            num_heads=4  # 减小到4
        ).to(device)
        
        optimizer = torch.optim.AdamW(final_model.parameters(), lr=0.0008, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8
        )

        total_params = sum(p.numel() for p in final_model.parameters())
        print(f"\n模型参数:")
        print(f"  - 总参数量: {total_params:,}")

        # 记录训练和测试损失
        train_losses = []
        test_losses = []

        for epoch in range(1, num_epochs + 1):
            fold_maes = []
            fold_r2s = []
            
            # K-Fold 交叉验证
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices)):
                # 创建 fold 的数据集
                fold_train_dataset = [train_dataset[i] for i in train_idx]
                fold_val_dataset = [train_dataset[i] for i in val_idx]
                
                fold_train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=True)
                fold_val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False)
                
                # 训练当前 fold
                train_loss = train_epoch(final_model, fold_train_loader, optimizer, device)
                
                # 验证当前 fold
                val_r2, val_mae, val_rmse, _, _ = evaluate(final_model, fold_val_loader, device)
                
                fold_maes.append(val_mae)
                fold_r2s.append(val_r2)
            
            # 计算交叉验证平均指标
            cv_mae = np.mean(fold_maes)
            cv_r2 = np.mean(fold_r2s)
            cv_mae_std = np.std(fold_maes)
            
            # 学习率调度
            scheduler.step(cv_mae)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 计算测试集损失（仅用于绘图）
            test_loss = compute_loss(final_model, test_loader, device)
            train_losses.append(cv_mae)  # 使用交叉验证平均 MAE 作为训练损失
            test_losses.append(test_loss)
            
            # 打印进度
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}/{num_epochs}: "
                      f"CV MAE={cv_mae:.4f}±{cv_mae_std:.4f}, "
                      f"CV R²={cv_r2:.4f} | "
                      f"LR={current_lr:.6f}")
            
            # Early stopping (基于交叉验证平均 MAE)
            if cv_mae < best_cv_mae:
                best_cv_mae = cv_mae
                best_epoch = epoch
                patience_counter = 0
                # 保存最佳预训练模型
                best_model_path = MODEL_DIR / "qm9_pretrained.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': final_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cv_mae': cv_mae,
                    'cv_r2': cv_r2,
                    'cv_mae_std': cv_mae_std,
                }, best_model_path)
                if epoch % 5 == 0 or epoch == 1:
                    print(f"  → 最佳模型已保存！(CV MAE 改善至: {cv_mae:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n{'=' * 60}")
                    print(f"Early stopping 于 epoch {epoch}")
                    print(f"最佳交叉验证 MAE: {best_cv_mae:.4f}")
                    print(f"{'=' * 60}")
                    break

        # 加载最佳预训练模型
        print(f"\n加载最佳模型...")
        best_model_path = MODEL_DIR / "qm9_pretrained.pt"
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        final_model.load_state_dict(checkpoint['model_state_dict'])

        # 在测试集上评估
        print("\n" + "=" * 60)
        print("测试集最终评估")
        print("=" * 60)
        test_r2, test_mae, test_rmse, _, _ = evaluate(final_model, test_loader, device)

        print(f"\n最佳交叉验证性能:")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  CV R²:    {checkpoint['cv_r2']:.4f}")
        print(f"  CV MAE:   {checkpoint['cv_mae']:.4f}±{checkpoint['cv_mae_std']:.4f}")

        print(f"\n测试集性能:")
        print(f"  R²:    {test_r2:.4f}")
        print(f"  MAE:   {test_mae:.4f}")
        print(f"  RMSE:  {test_rmse:.4f}")
        
        # 计算过拟合程度
        overfitting_gap = test_mae - checkpoint['cv_mae']
        print(f"\n过拟合分析:")
        print(f"  MAE差距: {overfitting_gap:.4f} ({'改善' if overfitting_gap < 0.1 else '需要进一步优化'})")

        # 绘制损失曲线
        print("\n" + "=" * 60)
        print("绘制损失曲线")
        print("=" * 60)
        loss_plot_path = MODEL_DIR / "qm9_pretrain_loss_curve.png"
        plot_loss_curves(
            train_losses=train_losses,
            test_losses=test_losses,
            save_path=loss_plot_path,
            title="QM9 Pretraining: Train vs Test Loss"
        )

        print(f"\n✓ 预训练成功完成！")
        print(f"✓ 最佳模型已保存至: {best_model_path}")
        print(f"✓ 损失曲线已保存至: {loss_plot_path}")

        return True

    except Exception as e:
        print(f"\n✗ 预训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("QM9 预训练脚本")
    print("=" * 60)

    # 创建模型目录
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    # 检查数据集文件
    dataset_path = DATA_DIR / "qm9_prepared.joblib"

    if not dataset_path.exists():
        print(f"\n✗ 数据集文件不存在: {dataset_path}")
        print("\n请先运行以下命令生成 QM9 预处理数据集:")
        print("  python scripts/input_preparation/input_graph_preparation.py \\")
        sys.exit(1)

    print(f"\n✓ 找到数据集: {dataset_path}")

    # 开始预训练
    success = pretrain_on_qm9(
        dataset_path=dataset_path,
    )

    if success:
        print("\n" + "=" * 60)
        print("✓ 预训练完成！")
        print("=" * 60)
        print(f"\n生成的文件:")
        print(f"  {MODEL_DIR / 'qm9_pretrained.pt'} - 预训练模型检查点")
        print(f"  {MODEL_DIR / 'qm9_pretrain_loss_curve.png'} - 训练损失曲线")
    else:
        print("\n" + "=" * 60)
        print("✗ 预训练失败！")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

