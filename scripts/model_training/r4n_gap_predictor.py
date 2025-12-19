import pandas as pd
import numpy as np
from pathlib import Path
from src import R4NGapDataset
import torch
from torch_geometric.loader import DataLoader
from src import GapPredictionGNN, train_epoch, evaluate
from utils import metrics_to_df, scat_avp
from IPython.display import Image, display
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" 


def main():
    print("R4N+ Gap Predictor Launching...")
    MODEL_DIR.mkdir(exist_ok=True)
    print("=" * 50)

    print("Searching for CUDA device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 50)

    # 加载数据集
    print("Loading dataset...")
    csv_path = DATA_DIR / "r4n_dft.csv"
    dataset = R4NGapDataset(csv_path)

    if len(dataset) == 0:
        print("Error: No valid data found!")
        return
    else:
        print(f"Dataset loaded successfully: {len(dataset)} samples")

    # 首先划分出测试集 (20%)
    print("\nSplitting dataset: 80% train, 20% test...")
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # 测试集DataLoader
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # K折交叉验证设置
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    print(f"\nUsing {n_folds}-Fold Cross Validation during training")
    
    # 创建模型
    print("\nInitializing model...")
    sample_data = dataset[0]
    num_node_features = sample_data.x.shape[1]
    print(f"Node features: {num_node_features}")
    print(f"Model parameters: ", end="")
    
    model = GapPredictionGNN(
        num_node_features=num_node_features,
        hidden_channels=80,  # 增加到80
        num_global_features=4,
        num_heads=4
    ).to(device)
    
    print(f"{sum(p.numel() for p in model.parameters())}")
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0023,  # 略微降低学习率
        weight_decay=3e-5  # 增加正则化
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=25
    )
    
    # 训练模型
    print("\nStarting model training with cross-validation...")
    print("=" * 50)
    num_epochs = 300
    best_val_mae = float('inf')
    patience = 40
    patience_counter = 0
    
    train_indices = list(range(len(train_dataset)))
    
    for epoch in range(1, num_epochs + 1):
        # 每个epoch在训练集上做K折交叉验证
        fold_train_losses = []
        fold_val_metrics = {'r2': [], 'mae': [], 'rmse': []}
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_indices), 1):
            # 创建当前fold的数据集
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
            
            # 在当前fold上训练一个epoch
            train_loss = train_epoch(model, train_loader, optimizer, device)
            fold_train_losses.append(train_loss)
            
            # 在当前fold的验证集上评估
            val_r2, val_mae, val_rmse, _, _ = evaluate(model, val_loader, device)
            fold_val_metrics['r2'].append(val_r2)
            fold_val_metrics['mae'].append(val_mae)
            fold_val_metrics['rmse'].append(val_rmse)
        
        # 计算所有fold的平均指标
        avg_train_loss = np.mean(fold_train_losses)
        avg_val_r2 = np.mean(fold_val_metrics['r2'])
        avg_val_mae = np.mean(fold_val_metrics['mae'])
        avg_val_rmse = np.mean(fold_val_metrics['rmse'])
        
        scheduler.step(avg_val_mae)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: "
                  f"Train Loss={avg_train_loss:.4f} | "
                  f"Val R²={avg_val_r2:.4f}, MAE={avg_val_mae:.4f}, RMSE={avg_val_rmse:.4f}")
        
        # Early stopping（基于交叉验证的平均MAE）
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), MODEL_DIR / "gap_prediction_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    # 加载最佳模型
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(MODEL_DIR / "gap_prediction_best.pt"))
    
    # 在完整训练集和测试集上评估
    full_train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    train_r2, train_mae, train_rmse, train_pred, train_true = evaluate(model, full_train_loader, device)
    test_r2, test_mae, test_rmse, test_pred, test_true = evaluate(model, test_loader, device)

    print(f"\nTraining Set:")
    print(f"  R²:   {train_r2:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")


    print(f"\nTest Set:")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")


    # 使用functions.py中的函数计算指标
    metrics_df = metrics_to_df(
        train_true, train_pred,
        test_true, test_pred,
        model_name='GNN_Gap_Predictor'
    )
    print(f"\nMetrics Summary:")
    print(metrics_df.to_string(index=False))

    # 保存预测结果
    results_df = pd.DataFrame({
        'True_Gap': test_true,
        'Predicted_Gap': test_pred,
        'Absolute_Error': np.abs(test_true - test_pred)
    })

    output_path = DATA_DIR / "gap_predictions.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # 使用functions.py绘制预测值vs真实值散点图
    print("\nGenerating scatter plot...")
    gap_min = min(test_true.min(), test_pred.min(), train_true.min(), train_pred.min())
    gap_max = max(test_true.max(), test_pred.max(), train_true.max(), train_pred.max())

    # 计算合适的坐标轴范围 (向下/向上取整到1)
    axis_min = np.floor(gap_min)  # 向下取整到1
    axis_max = np.ceil(gap_max)   # 向上取整到1

    print(f"Gap range: {gap_min:.2f} - {gap_max:.2f} eV")
    print(f"Axis range: {axis_min:.2f} - {axis_max:.2f} eV")

    scat_avp(
        y_train=train_true,
        y_train_pred=train_pred,
        y_test=test_true,
        y_test_pred=test_pred,
        save_path=str(MODEL_DIR / "gap_prediction_scatter.png"),
        axis_min=axis_min,
        axis_max=axis_max,
        model_name='GNN Gap Predictor',
        title='HOMO-LUMO Gap Prediction',
        figsize=(8, 8),
        dpi=600
    )

    print(f"\nScatter plot saved to: {MODEL_DIR / 'gap_prediction_scatter.png'}")

    # 显示图片
    display(Image(str(MODEL_DIR / "gap_prediction_scatter.png")))

if __name__ == "__main__":
    main()