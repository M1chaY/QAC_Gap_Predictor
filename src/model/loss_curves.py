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
    figsize: tuple = (8.5, 8), 
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
    
    # 计算最小值信息
    min_train_idx = np.argmin(train_losses)
    min_test_idx = np.argmin(test_losses)
    min_train_val = train_losses[min_train_idx]
    min_test_val = test_losses[min_test_idx]
    
    # 绘制损失曲线（图例中包含最小值信息）
    ax.plot(
        epochs, train_losses, 'b-', linewidth=1, alpha=0.8,
        label=f'Train (Min: {min_train_val:.4f} @ Epoch {min_train_idx + 1})'
    )
    ax.plot(
        epochs, test_losses, 'r-', linewidth=1, alpha=0.8,
        label=f'Val (Min: {min_test_val:.4f} @ Epoch {min_test_idx + 1})'
    )
    
    # 标注最小值点
    ax.plot(min_train_idx + 1, min_train_val, 'bo', markersize=8)
    ax.plot(min_test_idx + 1, min_test_val, 'ro', markersize=8)
    
    # 设置标签和标题
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss (MAE)', fontsize=20)
    ax.set_title(title, fontsize=24, pad=10)
    
    # 设置图例和网格
    ax.legend(frameon=False, fontsize=14, loc='upper right')
    ax.grid(False)
    
    # 设置刻度样式
    ax.tick_params(
        axis='both', which='both', 
        length=3, width=1, 
        colors='black', labelsize=18
    )
    
    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
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
