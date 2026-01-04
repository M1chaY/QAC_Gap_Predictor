"""
训练可视化工具

提供训练过程可视化功能。
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: List[float], 
    test_losses: List[float], 
    save_path: Optional[str] = None, 
    title: str = "Training and Test Loss Curves", 
    figsize: tuple = (10, 6), 
    dpi: int = 600
) -> None:
    """
    绘制训练集和测试集的损失曲线。
    
    Args:
        train_losses: 训练集损失列表（每个epoch的平均损失）
        test_losses: 测试集损失列表（每个epoch的平均损失）
        save_path: 保存路径，为None则显示图像
        title: 图表标题
        figsize: 图形尺寸
        dpi: 图形分辨率
    """
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失曲线
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
    
    # 标注最小值
    min_train_idx = np.argmin(train_losses)
    min_test_idx = np.argmin(test_losses)
    
    ax.plot(min_train_idx + 1, train_losses[min_train_idx], 'bo', markersize=8)
    ax.plot(min_test_idx + 1, test_losses[min_test_idx], 'ro', markersize=8)
    
    ax.text(
        min_train_idx + 1, train_losses[min_train_idx], 
        f'  Min: {train_losses[min_train_idx]:.4f}',
        fontsize=14, verticalalignment='bottom'
    )
    ax.text(
        min_test_idx + 1, test_losses[min_test_idx], 
        f'  Min: {test_losses[min_test_idx]:.4f}',
        fontsize=14, verticalalignment='top'
    )
    
    # 设置标签和标题
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss (MAE)', fontsize=20)
    ax.set_title(title, fontsize=24, pad=10)
    
    # 设置图例和网格
    ax.legend(frameon=False, fontsize=20, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 设置刻度样式
    ax.tick_params(
        axis='both', which='both', 
        length=5, width=2, 
        colors='black', labelsize=16
    )
    
    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, transparent=False)
        print(f"\nLoss curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_prediction_scatter(
    targets: np.ndarray,
    predictions: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Predicted vs Actual Gap",
    figsize: tuple = (8, 8),
    dpi: int = 600
) -> None:
    """
    绘制预测值与实际值的散点图。
    
    Args:
        targets: 实际值数组
        predictions: 预测值数组
        save_path: 保存路径，为None则显示图像
        title: 图表标题
        figsize: 图形尺寸
        dpi: 图形分辨率
    """
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # 散点图
    ax.scatter(targets, predictions, alpha=0.5, s=20, c='blue', edgecolors='none')
    
    # 对角线
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    
    ax.set_xlabel('Actual Gap (eV)', fontsize=16)
    ax.set_ylabel('Predicted Gap (eV)', fontsize=16)
    ax.set_title(title, fontsize=20, pad=10)
    
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, transparent=False)
        print(f"\nScatter plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)
