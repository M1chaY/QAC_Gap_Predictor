"""
增强版微调工具（仅用于 enhanced 版本）。

本模块承载 gnn_finetune_onqac_enhanced.py 的通用辅助函数，
避免脚本文件过长，且不影响原有标准版本流程。
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch_geometric.loader import DataLoader


def evaluate_test_set_with_state(
    create_model_fn: Callable,
    evaluate_fn: Callable,
    checkpoint: dict,
    best_state: dict,
    test_data: list,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, float, bool]:
    """加载指定状态并在测试集评估。"""
    model = create_model_fn(checkpoint, device)
    model.load_state_dict(best_state)
    model.eval()

    has_test_set = len(test_data) > 0
    if has_test_set:
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        test_r2, test_mae, test_rmse, _, _ = evaluate_fn(model, test_loader, device)
        print("\nTest set performance:")
        print(f"  R2:   {test_r2:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        return test_r2, test_mae, test_rmse, has_test_set

    test_r2 = float("nan")
    test_mae = float("nan")
    test_rmse = float("nan")
    print("\nWarning: test set is empty, skip test evaluation.")
    return test_r2, test_mae, test_rmse, has_test_set


def aggregate_fold_histories(
    fold_train_histories: list,
    fold_val_histories: list,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """聚合各折历史曲线（早停长度不一致时用NaN对齐）。"""
    max_fold_epochs = max(len(hist) for hist in fold_train_histories)
    fold_train_matrix = np.full((n_folds, max_fold_epochs), np.nan, dtype=float)
    fold_val_matrix = np.full((n_folds, max_fold_epochs), np.nan, dtype=float)
    for i in range(n_folds):
        fold_train_matrix[i, : len(fold_train_histories[i])] = fold_train_histories[i]
        fold_val_matrix[i, : len(fold_val_histories[i])] = fold_val_histories[i]

    return (
        np.nanmean(fold_train_matrix, axis=0),
        np.nanmean(fold_val_matrix, axis=0),
        fold_val_matrix,
    )


def build_finetune_info(
    num_samples: int,
    n_folds: int,
    has_test_set: bool,
    freeze_strategy: str,
    trainable_stats: dict,
    lr: float,
    weight_decay: float,
    weight_decay_source: str,
    coord_noise_std: float,
    node_mask_ratio: float,
    pretrained_path: Path,
) -> dict:
    """构建并返回保存到checkpoint的微调信息。"""
    return {
        "num_samples": num_samples,
        "n_folds": n_folds,
        "has_test_set": has_test_set,
        "freeze_strategy": freeze_strategy,
        "resolved_strategy": trainable_stats["strategy"],
        "strategy_desc": trainable_stats["strategy_desc"],
        "trainable_params": trainable_stats["trainable_params"],
        "total_params": trainable_stats["total_params"],
        "trainable_ratio": trainable_stats["trainable_ratio"],
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "weight_decay_source": weight_decay_source,
        "coord_noise_std": coord_noise_std,
        "node_mask_ratio": node_mask_ratio,
        # 记录来源checkpoint，便于溯源复现实验
        "pretrained_from": str(pretrained_path),
    }


def save_finetune_outputs(
    params_dir: Path,
    finetuned_path: Path,
    history_path: Path,
    best_state: dict,
    model_config: dict,
    best_epoch: int,
    best_avg_mae: float,
    test_r2: float,
    test_mae: float,
    test_rmse: float,
    dataset,
    finetune_info: dict,
    epoch_train_losses: np.ndarray,
    epoch_val_losses: np.ndarray,
    fold_val_losses: np.ndarray,
    fold_best_maes: list,
    fold_best_epochs: list,
    best_fold_idx: int,
    has_test_set: bool,
) -> None:
    """保存微调模型与训练历史。"""
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": model_config,
            "best_epoch": best_epoch,
            "cv_avg_mae": best_avg_mae,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "finetune_info": finetune_info,
            "dataset_info": {
                "num_node_features": dataset.num_node_features,
                "num_edge_features": dataset.num_edge_features,
                "num_global_features": dataset.num_global_features,
            },
        },
        finetuned_path,
    )
    print(f"\nFinetuned model saved: {finetuned_path}")

    params_dir.mkdir(exist_ok=True, parents=True)
    np.savez(
        history_path,
        epoch_train_losses=epoch_train_losses,
        epoch_val_losses=epoch_val_losses,
        fold_val_losses=fold_val_losses,
        fold_best_maes=np.array(fold_best_maes),
        fold_best_epochs=np.array(fold_best_epochs),
        selected_best_fold=best_fold_idx + 1,
        best_epoch=best_epoch,
        cv_avg_mae=best_avg_mae,
        has_test_set=has_test_set,
        test_mae=test_mae,
        test_r2=test_r2,
        test_rmse=test_rmse,
    )
    print(f"Training history saved: {history_path}")
