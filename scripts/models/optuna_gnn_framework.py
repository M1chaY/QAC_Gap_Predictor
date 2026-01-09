"""
Optuna GNN超参数搜索框架

使用Optuna进行GNN模型结构和超参数的自动搜索。

搜索策略：
1. 数据集按70:15:15划分（训练:验证:测试），固定随机种子
2. 基础倍率BASE_UNIT=16，每层维度为倍率的1-8倍
3. GAT层数1-3层，MLP层数2-5层，注意力头2/4/8
4. 每个结构训练200次迭代，30次耐心值早停
5. 以验证集MAE作为优化目标
6. 保存最佳结构为MD报告和JSON格式
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import optuna
import torch
from torch_geometric.loader import DataLoader

from src import (
    PreparedGraphDataset,
    GapPredictionGNN,
    train_epoch,
    evaluate,
    set_seed,
    split_dataset,
    save_optuna_results,
    setup_training_logger,
    log_trial_start,
    log_epoch,
    log_trial_end,
    log_search_summary,
)
from src.model.gap_gnn import BASE_UNIT

# 忽略torch-scatter警告
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# 子目录路径
LOG_DIR = MODEL_DIR / "log"
PARAMS_DIR = MODEL_DIR / "params"

# 固定随机种子
RANDOM_SEED = 42

# 搜索空间常量
MIN_MULTIPLIER = 1  # 最小倍率
MAX_MULTIPLIER = 8  # 最大倍率
MIN_GAT_LAYERS = 1
MAX_GAT_LAYERS = 3
MIN_MLP_LAYERS = 2
MAX_MLP_LAYERS = 5
VALID_HEADS = [2, 4, 8]


def sample_layer_dims(
    trial: optuna.Trial, 
    layer_type: str, 
    num_layers: int, 
    num_heads: int
) -> List[int]:
    """
    采样每层的维度。
    
    Args:
        trial: Optuna trial对象
        layer_type: 层类型（'gat' 或 'mlp'）
        num_layers: 层数
        num_heads: 注意力头数（仅GAT层需要，用于确保可整除）
        
    Returns:
        List[int]: 每层的维度列表
    """
    dims = []
    for i in range(num_layers):
        # 采样倍率
        multiplier = trial.suggest_int(
            f'{layer_type}_layer{i}_multiplier', 
            MIN_MULTIPLIER, 
            MAX_MULTIPLIER
        )
        dim = BASE_UNIT * multiplier
        
        # GAT层需要确保可被num_heads整除
        if layer_type == 'gat' and dim % num_heads != 0:
            # 向上取整到可整除的值
            dim = ((dim // num_heads) + 1) * num_heads
        
        dims.append(dim)
    
    return dims


def create_objective(dataset, device, logger, num_epochs=200, patience=30, batch_size=64):
    """
    创建Optuna目标函数。
    
    Args:
        dataset: 数据集对象
        device: 计算设备
        logger: 日志记录器
        num_epochs: 每个trial的最大训练轮数
        patience: 每个trial的早停耐心值
        batch_size: 批次大小
        
    Returns:
        callable: Optuna目标函数
        tuple: (train_data, val_data, test_data)
    """
    # 划分数据集（70:15:15）
    train_data, val_data, test_data = split_dataset(
        dataset, train_ratio=0.7, val_ratio=0.15, seed=RANDOM_SEED
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples (70%)")
    print(f"  Val:   {len(val_data)} samples (15%)")
    print(f"  Test:  {len(test_data)} samples (15%)")
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna目标函数：返回验证集MAE。"""
        trial_number = trial.number + 1
        
        # 结构超参数
        num_gat_layers = trial.suggest_int(
            'num_gat_layers', MIN_GAT_LAYERS, MAX_GAT_LAYERS
        )
        num_mlp_layers = trial.suggest_int(
            'num_mlp_layers', MIN_MLP_LAYERS, MAX_MLP_LAYERS
        )
        num_heads = trial.suggest_categorical('num_heads', VALID_HEADS)
        
        # 每层维度采样
        gat_dims = sample_layer_dims(trial, 'gat', num_gat_layers, num_heads)
        mlp_dims = sample_layer_dims(trial, 'mlp', num_mlp_layers, num_heads)
        
        # 训练超参数
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # 记录trial开始和超参数
        params = {
            'gat_dims': gat_dims,
            'mlp_dims': mlp_dims,
            'num_heads': num_heads,
            'dropout': dropout,
            'lr': lr,
            'weight_decay': weight_decay
        }
        log_trial_start(logger, trial_number, params)
        
        # 构建模型
        model = GapPredictionGNN(
            num_node_features=dataset.num_node_features,
            gat_dims=gat_dims,
            mlp_dims=mlp_dims,
            num_global_features=dataset.num_global_features,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)
        
        # 优化器和学习率调度
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # 训练循环
        best_val_mae = float('inf')
        patience_counter = 0
        final_epoch = 0
        
        for epoch in range(1, num_epochs + 1):
            final_epoch = epoch
            
            # 训练
            train_loss = train_epoch(model, train_loader, optimizer, device)
            
            # 验证
            val_r2, val_mae, val_rmse, _, _ = evaluate(model, val_loader, device)
            
            # 学习率调度
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_mae)
            
            # 记录epoch
            is_best = val_mae < best_val_mae
            log_epoch(logger, trial_number, epoch, train_loss, val_mae, val_r2, 
                     current_lr, is_best)
            
            # 早停检查
            if is_best:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log_trial_end(logger, trial_number, best_val_mae, 
                                 final_epoch, "EARLY_STOPPED")
                    break
            
            # Optuna剪枝
            trial.report(val_mae, epoch)
            if trial.should_prune():
                log_trial_end(logger, trial_number, best_val_mae, 
                             final_epoch, "PRUNED")
                raise optuna.TrialPruned()
        else:
            # 正常完成所有epochs
            log_trial_end(logger, trial_number, best_val_mae, 
                         final_epoch, "COMPLETED")
        
        return best_val_mae
    
    return objective, (train_data, val_data, test_data)


def extract_best_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    从trial中提取最佳超参数配置。
    
    Args:
        trial: Optuna trial对象
        
    Returns:
        Dict: 包含gat_dims, mlp_dims等超参数的字典
    """
    num_gat_layers = trial.params['num_gat_layers']
    num_mlp_layers = trial.params['num_mlp_layers']
    num_heads = trial.params['num_heads']
    
    gat_dims = []
    for i in range(num_gat_layers):
        multiplier = trial.params[f'gat_layer{i}_multiplier']
        dim = BASE_UNIT * multiplier
        if dim % num_heads != 0:
            dim = ((dim // num_heads) + 1) * num_heads
        gat_dims.append(dim)
    
    mlp_dims = []
    for i in range(num_mlp_layers):
        multiplier = trial.params[f'mlp_layer{i}_multiplier']
        mlp_dims.append(BASE_UNIT * multiplier)
    
    return {
        'gat_dims': gat_dims,
        'mlp_dims': mlp_dims,
        'num_heads': num_heads,
        'dropout': trial.params['dropout'],
        'lr': trial.params['lr'],
        'weight_decay': trial.params['weight_decay']
    }


def save_intermediate_results(
    study: optuna.Study,
    dataset_info: Dict[str, Any],
    search_info: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    保存当前最佳搜索结果（中间结果）。
    
    Args:
        study: Optuna study对象
        dataset_info: 数据集信息字典
        search_info: 搜索配置信息字典
        output_dir: 输出目录
    """
    if len(study.trials) == 0:
        return
    
    # 获取最佳trial
    try:
        best_trial = study.best_trial
        best_value = study.best_value
    except ValueError:
        # 没有完成的trial
        return
    
    # 提取最佳参数
    best_params = extract_best_params(best_trial)
    
    # 统计信息
    n_completed = len([t for t in study.trials 
                       if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials 
                    if t.state == optuna.trial.TrialState.PRUNED])
    
    # 构建结果（不含测试集评估，标记为中间结果）
    results = {
        "search_info": search_info,
        "dataset_info": dataset_info,
        "best_params": best_params,
        "best_performance": {
            "val_mae": float(best_value),
            "test_r2": None,
            "test_mae": None,
            "test_rmse": None,
            "note": "Intermediate result - test evaluation pending"
        },
        "study_statistics": {
            "n_completed_trials": n_completed,
            "n_pruned_trials": n_pruned,
            "best_trial_number": best_trial.number + 1,
            "model_parameters": None
        }
    }
    
    # 保存结果
    save_optuna_results(results, output_dir)


def run_hyperparameter_search(
    dataset_path: Path,
    n_trials: int = 200,
    num_epochs: int = 200,
    patience: int = 30,
    batch_size: int = 64
) -> dict:
    """
    运行超参数搜索。
    
    Args:
        dataset_path: 数据集路径
        n_trials: 搜索试验次数
        num_epochs: 每个trial的最大训练轮数
        patience: 早停耐心值
        batch_size: 批次大小
        
    Returns:
        tuple: (结果字典, logger对象)
    """
    # 设置随机种子确保可重复性
    set_seed(RANDOM_SEED)
    
    # 设置日志记录器
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    logger = setup_training_logger(LOG_DIR, log_name="optuna_search")
    
    print("\n" + "=" * 60)
    print("Loading dataset")
    print("=" * 60)
    
    dataset = PreparedGraphDataset(dataset_path)
    
    print(f"\nDataset info:")
    print(f"  Samples: {len(dataset)}")
    print(f"  Node features: {dataset.num_node_features}")
    print(f"  Edge features: {dataset.num_edge_features}")
    print(f"  Global features: {dataset.num_global_features}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # 显示搜索空间信息
    print(f"\nSearch space:")
    print(f"  Base unit: {BASE_UNIT}")
    print(f"  Layer dim range: {BASE_UNIT * MIN_MULTIPLIER} - {BASE_UNIT * MAX_MULTIPLIER}")
    print(f"  GAT layers: {MIN_GAT_LAYERS} - {MAX_GAT_LAYERS}")
    print(f"  MLP layers: {MIN_MLP_LAYERS} - {MAX_MLP_LAYERS}")
    print(f"  Attention heads: {VALID_HEADS}")
    
    # 创建目标函数（传入logger）
    objective, (train_data, val_data, test_data) = create_objective(
        dataset, device, logger, num_epochs, patience, batch_size
    )
    
    # 准备数据集和搜索配置信息（用于中间保存）
    dataset_info = {
        "total_samples": len(dataset),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "num_node_features": dataset.num_node_features,
        "num_edge_features": dataset.num_edge_features,
        "num_global_features": dataset.num_global_features
    }
    
    search_info = {
        "n_trials": n_trials,
        "num_epochs_per_trial": num_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "random_seed": RANDOM_SEED,
        "base_unit": BASE_UNIT,
        "multiplier_range": [MIN_MULTIPLIER, MAX_MULTIPLIER],
        "dataset_path": str(dataset_path),
        "timestamp": datetime.now().isoformat()
    }
    
    # 创建Optuna研究
    print("\n" + "=" * 60)
    print("Starting hyperparameter search")
    print("=" * 60)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    )
    
    # 定义搜索过程的回调（早停 + 自动保存最优结果）
    class SearchCallback:
        """搜索回调：早停检测 + 发现更优结果时自动保存。"""
        
        def __init__(self, patience: int = 50):
            self.patience = patience
            self.best_value = float('inf')
            self.patience_counter = 0
        
        def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
            if trial.value is None:
                return
            
            if trial.value < self.best_value:
                self.best_value = trial.value
                self.patience_counter = 0
                
                # 发现更优结果，保存中间结果
                save_intermediate_results(
                    study, dataset_info, search_info, PARAMS_DIR
                )
                print(f"  [Auto-saved] Best config updated (Val MAE: {trial.value:.4f})")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f"\nSearch early stopped: No improvement for {self.patience} trials")
                study.stop()
    
    # 搜索早停设置
    search_callback = SearchCallback(patience=30)
    
    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=[search_callback]
    )
    
    # 提取最佳参数（使用辅助函数）
    best_trial = study.best_trial
    best_value = study.best_value
    best_params = extract_best_params(best_trial)
    
    print("\n" + "=" * 60)
    print("Search completed")
    print("=" * 60)
    print(f"\nBest validation MAE: {best_value:.4f}")
    print(f"\nBest hyperparameters:")
    print(f"  GAT dims: {best_params['gat_dims']}")
    print(f"  MLP dims: {best_params['mlp_dims']}")
    print(f"  Num heads: {best_params['num_heads']}")
    print(f"  Dropout: {best_params['dropout']:.1f}")
    print(f"  LR: {best_params['lr']:.6f}")
    print(f"  Weight decay: {best_params['weight_decay']:.6f}")
    
    # 在测试集上评估最佳模型
    print("\n" + "=" * 60)
    print("Evaluating best model on test set")
    print("=" * 60)
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # 重新训练最佳模型
    best_model = GapPredictionGNN(
        num_node_features=dataset.num_node_features,
        gat_dims=best_params['gat_dims'],
        mlp_dims=best_params['mlp_dims'],
        num_global_features=dataset.num_global_features,
        num_heads=best_params['num_heads'],
        dropout=best_params['dropout']
    ).to(device)
    
    print(f"\nModel parameters: {best_model.count_parameters():,}")
    
    optimizer = torch.optim.AdamW(
        best_model.parameters(), 
        lr=best_params['lr'], 
        weight_decay=best_params['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    
    best_val_mae = float('inf')
    patience_counter = 0
    best_state = None
    
    for _ in range(1, num_epochs + 1):
        _ = train_epoch(best_model, train_loader, optimizer, device)
        _, val_mae, _, _, _ = evaluate(best_model, val_loader, device)
        scheduler.step(val_mae)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            # 保存最佳状态
            best_state = {k: v.cpu().clone() for k, v in best_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # 加载最佳状态并评估测试集
    if best_state is not None:
        best_model.load_state_dict(best_state)
    test_r2, test_mae, test_rmse, _, _ = evaluate(best_model, test_loader, device)
    
    print(f"\nTest set performance:")
    print(f"  R2:   {test_r2:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    
    # 构建最终结果（复用已准备的信息）
    search_info["timestamp"] = datetime.now().isoformat()  # 更新时间戳
    
    results = {
        "search_info": search_info,
        "dataset_info": dataset_info,
        "best_params": best_params,
        "best_performance": {
            "val_mae": float(best_value),
            "test_r2": float(test_r2),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse)
        },
        "study_statistics": {
            "n_completed_trials": len(study.trials),
            "n_pruned_trials": len([
                t for t in study.trials 
                if t.state == optuna.trial.TrialState.PRUNED
            ]),
            "model_parameters": best_model.count_parameters()
        }
    }
    
    # 记录搜索总结到日志
    test_metrics = {
        'R2': test_r2,
        'MAE': test_mae,
        'RMSE': test_rmse
    }
    log_search_summary(
        logger, best_params, best_value, test_metrics,
        results['study_statistics']['n_completed_trials'],
        results['study_statistics']['n_pruned_trials']
    )
    
    return results, logger


def main():
    """主函数。"""
    print("=" * 60)
    print("Optuna GNN Hyperparameter Search Framework")
    print("=" * 60)
    
    # 创建输出目录
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    
    # 检查数据集
    dataset_path = DATA_DIR / "qm9_prepared.joblib"
    
    if not dataset_path.exists():
        print(f"\nDataset not found: {dataset_path}")
        print("\nPlease run the following command first:")
        print("  python scripts/input_preparation/input_graph_preparation.py")
        sys.exit(1)
    
    print(f"\nDataset found: {dataset_path}")
    
    # 运行超参数搜索
    results, logger = run_hyperparameter_search(
        dataset_path=dataset_path,
        n_trials=200,
        num_epochs=200,
        patience=30,
        batch_size=64
    )
    
    # 保存结果
    PARAMS_DIR.mkdir(exist_ok=True, parents=True)
    json_path, md_path = save_optuna_results(results, PARAMS_DIR)
    print(f"\nJSON config saved: {json_path}")
    print(f"Markdown report saved: {md_path}")
    
    # 获取日志文件路径
    log_file = logger.handlers[0].baseFilename if logger.handlers else "N/A"
    
    print("\n" + "=" * 60)
    print("Hyperparameter search completed!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  {json_path} - Best configuration")
    print(f"  {md_path} - Search report")
    print(f"  {log_file} - Training log")


if __name__ == "__main__":
    main()
