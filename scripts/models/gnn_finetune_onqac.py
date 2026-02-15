"""
QAC GNN微调脚本

加载QM9预训练模型，使用QAC数据集进行微调。

策略：
1. 读取预训练模型和配置
2. 加载QAC数据集，选取指定数量样本用于训练
3. 五折交叉验证训练，以平均MAE作为评估指标
4. 剩余样本作为测试集
5. 可变学习率（ReduceLROnPlateau）
6. 保存训练/测试loss用于后续绘图
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from src import (
    PreparedGraphDataset,
    GapPredictionGNN,
    train_epoch,
    evaluate,
    set_seed
)

# 忽略torch-scatter警告
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
PARAMS_DIR = MODEL_DIR / "params"


def load_pretrained_checkpoint(model_path: Path, device: torch.device):
    """
    加载预训练模型检查点。
    
    Args:
        model_path: 预训练模型路径
        device: 计算设备
        
    Returns:
        dict: 检查点字典
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    
    print(f"Loading pretrained model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Val MAE: {checkpoint['val_mae']:.4f}")
    
    return checkpoint


def create_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """
    从检查点创建并初始化模型。
    
    Args:
        checkpoint: 模型检查点
        device: 计算设备
        
    Returns:
        GapPredictionGNN: 初始化后的模型
    """
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
    
    return model


def finetune_on_qac(
    pretrained_path: Path,
    dataset_path: Path,
    num_samples: int = 50,
    n_folds: int = 5,
    num_epochs: int = 200,
    patience: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    seed: int = 42
) -> bool:
    """
    使用QAC数据集进行五折交叉验证微调。
    
    Args:
        pretrained_path: 预训练模型路径
        dataset_path: QAC数据集路径
        num_samples: 用于训练的样本数量
        n_folds: 交叉验证折数
        num_epochs: 最大训练轮数
        patience: 早停耐心值
        batch_size: 批次大小
        lr: 初始学习率
        seed: 随机种子
        
    Returns:
        bool: 是否成功
    """
    try:
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练检查点
        print("\n" + "=" * 60)
        print("Loading pretrained model")
        print("=" * 60)
        
        checkpoint = load_pretrained_checkpoint(pretrained_path, device)
        model_config = checkpoint['model_config']
        
        print(f"\nModel config:")
        print(f"  GAT dims: {model_config['gat_dims']}")
        print(f"  MLP dims: {model_config['mlp_dims']}")
        
        # 加载QAC数据集
        print("\n" + "=" * 60)
        print("Loading QAC dataset")
        print("=" * 60)
        
        dataset = PreparedGraphDataset(dataset_path)
        print(f"\nDataset info:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Node features: {dataset.num_node_features}")
        print(f"  Global features: {dataset.num_global_features}")
        
        # 划分训练集和测试集
        if num_samples > len(dataset):
            print(f"Warning: requested {num_samples}, only {len(dataset)} available")
            num_samples = len(dataset)
        
        all_indices = np.random.permutation(len(dataset))
        train_indices = all_indices[:num_samples]
        test_indices = all_indices[num_samples:]
        
        train_data = [dataset[i] for i in train_indices]
        test_data = [dataset[i] for i in test_indices]
        
        print(f"\nData split:")
        print(f"  Train (for CV): {len(train_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        print(f"\nDevice: {device}")
        
        # 五折交叉验证 - 单一模型，每个epoch遍历5折训练
        print("\n" + "=" * 60)
        print(f"{n_folds}-Fold Cross Validation Finetuning")
        print("=" * 60)
        print(f"Learning rate: {lr}")
        print(f"Epochs: {num_epochs}, Patience: {patience}\n")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        weight_decay = model_config.get('weight_decay', 1e-4)
        
        # 准备每折的数据加载器
        fold_splits = list(kf.split(train_data))
        fold_loaders = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            fold_train = [train_data[i] for i in train_idx]
            fold_val = [train_data[i] for i in val_idx]
            train_loader = DataLoader(fold_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(fold_val, batch_size=batch_size, shuffle=False)
            fold_loaders.append((train_loader, val_loader))
            print(f"Fold {fold_idx + 1}: Train {len(fold_train)}, Val {len(fold_val)}")
        
        # 创建单一模型
        model = create_model_from_checkpoint(checkpoint, device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
        )
        
        print(f"\nModel parameters: {model.count_parameters():,}")
        
        # 训练历史记录
        epoch_train_losses = []  # 每个epoch的平均训练loss
        epoch_val_losses = []    # 每个epoch的平均验证MAE
        fold_val_losses = [[] for _ in range(n_folds)]  # 每折每个epoch的验证MAE
        
        best_avg_mae = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state = None
        min_delta = 1e-4  # 最小改进阈值
        
        print("\nTraining started...\n")
        
        for epoch in range(1, num_epochs + 1):
            epoch_fold_train = []
            epoch_fold_val = []
            
            # 每个epoch内遍历5折进行训练
            for fold_idx in range(n_folds):
                train_loader, val_loader = fold_loaders[fold_idx]
                
                # 用当前折的训练数据训练模型
                train_loss = train_epoch(model, train_loader, optimizer, device)
                epoch_fold_train.append(train_loss)
            
            # 训练完5折后，用每折的验证集评估
            for fold_idx in range(n_folds):
                _, val_loader = fold_loaders[fold_idx]
                _, val_mae, _, _, _ = evaluate(model, val_loader, device)
                epoch_fold_val.append(val_mae)
                fold_val_losses[fold_idx].append(val_mae)
            
            # 计算平均loss
            avg_train_loss = np.mean(epoch_fold_train)
            avg_val_mae = np.mean(epoch_fold_val)
            
            epoch_train_losses.append(avg_train_loss)
            epoch_val_losses.append(avg_val_mae)
            
            # 更新学习率调度器（基于平均MAE）
            scheduler.step(avg_val_mae)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 打印进度
            is_new_best = avg_val_mae < best_avg_mae - min_delta
            if epoch % 10 == 0 or epoch == 1 or is_new_best:
                fold_maes = ", ".join([f"{mae:.4f}" for mae in epoch_fold_val])
                best_marker = " *NEW BEST*" if is_new_best else ""
                print(f"Epoch {epoch:04d}/{num_epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, "
                      f"Avg Val MAE={avg_val_mae:.4f} | "
                      f"LR={current_lr:.2e}{best_marker}")
                print(f"         Fold MAEs: [{fold_maes}]")
            
            # 早停检查（基于平均MAE，考虑最小改进阈值）
            if avg_val_mae < best_avg_mae - min_delta:
                best_avg_mae = avg_val_mae
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"  No improvement > {min_delta} for {patience} epochs")
                    break
        
        print(f"\nBest Avg Val MAE: {best_avg_mae:.4f} at epoch {best_epoch}")
        
        # 加载最佳模型进行测试
        print("\n" + "=" * 60)
        print("Test set evaluation")
        print("=" * 60)
        
        model.load_state_dict(best_state)
        model.eval()
        
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        test_r2, test_mae, test_rmse, preds, targets = evaluate(
            model, test_loader, device
        )
        
        print(f"\nTest set performance:")
        print(f"  R2:   {test_r2:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        
        # 保存微调后模型
        finetuned_path = MODEL_DIR / "qac_finetuned.pt"
        torch.save({
            'model_state_dict': best_state,
            'model_config': model_config,
            'best_epoch': best_epoch,
            'cv_avg_mae': best_avg_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'finetune_info': {
                'num_samples': num_samples,
                'n_folds': n_folds,
                'learning_rate': lr,
                'pretrained_from': str(pretrained_path)
            },
            'dataset_info': {
                'num_node_features': dataset.num_node_features,
                'num_edge_features': dataset.num_edge_features,
                'num_global_features': dataset.num_global_features
            }
        }, finetuned_path)
        print(f"\nFinetuned model saved: {finetuned_path}")
        
        # 保存训练历史（用于绘图）
        PARAMS_DIR.mkdir(exist_ok=True, parents=True)
        history_path = PARAMS_DIR / "qac_finetune_history.npz"
        
        np.savez(
            history_path,
            epoch_train_losses=np.array(epoch_train_losses),
            epoch_val_losses=np.array(epoch_val_losses),
            fold_val_losses=np.array(fold_val_losses),
            best_epoch=best_epoch,
            cv_avg_mae=best_avg_mae,
            test_mae=test_mae,
            test_r2=test_r2,
            test_rmse=test_rmse
        )
        print(f"Training history saved: {history_path}")
        
        return True
        
    except Exception as e:
        print(f"\nFinetuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数。"""
    print("=" * 60)
    print("QAC GNN Finetuning Script (5-Fold CV)")
    print("=" * 60)
    
    pretrained_path = MODEL_DIR / "qm9_pretrained.pt"
    dataset_path = DATA_DIR / "qac_prepared.joblib"
    
    # 检查文件存在性
    if not pretrained_path.exists():
        print(f"Error: Pretrained model not found: {pretrained_path}")
        sys.exit(1)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # 执行微调（使用100条数据进行5折交叉验证，24条作为测试集）
    success = finetune_on_qac(
        pretrained_path=pretrained_path,
        dataset_path=dataset_path,
        num_samples=100,
        n_folds=5,
        num_epochs=1000,
        patience=50,
        batch_size=8,
        lr=0.001531,
        seed=42
    )
    
    if success:
        print("\n" + "=" * 60)
        print("Finetuning completed successfully")
        print("=" * 60)
    else:
        print("\nFinetuning failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
