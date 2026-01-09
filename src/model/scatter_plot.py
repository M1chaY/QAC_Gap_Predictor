"""
实际值vs预测值散点图

绘制训练集和测试集的预测效果对比图。
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_actual_vs_predicted(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    save_path: str,
    axis_min: float = 0,
    axis_max: float = 220,
    model_name: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8.5, 8),
    dpi: int = 600
) -> None:
    """
    绘制实际值vs预测值对比图。

    Args:
        y_train: 训练集实际值
        y_train_pred: 训练集预测值
        y_test: 测试集实际值
        y_test_pred: 测试集预测值
        save_path: 图片保存路径
        axis_min: 坐标轴最小值
        axis_max: 坐标轴最大值
        model_name: 模型名称
        title: 图表标题，为None则使用model_name
        figsize: 图形尺寸
        dpi: 图形分辨率
    """
    if title is None:
        title = f'{model_name}'

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 绘制散点图
    ax.scatter(y_train, y_train_pred, color='blue', label='Train', s=30, alpha=0.3)
    ax.scatter(y_test, y_test_pred, color='red', label='Test', s=30, alpha=0.3)
    ax.plot(
        [axis_min, axis_max], [axis_min, axis_max], 
        'k--', lw=1, label='Perfect Prediction'
    )

    # 设置坐标轴范围和刻度
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    
    # 固定6个主刻度，每两个主刻度之间1个副刻度
    data_range = axis_max - axis_min
    major_interval = data_range / 5  # 6个刻度需要5个间隔
    minor_interval = major_interval / 2
    
    major_ticks = np.linspace(axis_min, axis_max, 6)
    minor_ticks = np.arange(axis_min + minor_interval, axis_max, major_interval)
    
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(minor_ticks, minor=True)

    # 设置标签和标题
    ax.set_xlabel('Actual Values', fontsize=20)
    ax.set_ylabel('Predicted Values', fontsize=20)
    ax.set_title(title, fontsize=24, pad=10)

    # 设置图例和刻度样式
    ax.legend(frameon=False, fontsize=20, loc='upper left')
    ax.tick_params(
        axis='both', which='major', 
        length=5, width=1, 
        colors='black', labelsize=18
    )
    ax.tick_params(
        axis='both', which='minor', 
        length=3, width=0.8, 
        colors='black'
    )

    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # 保存图形
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, transparent=False)
    plt.close(fig)
