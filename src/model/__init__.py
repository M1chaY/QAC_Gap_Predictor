"""
模型模块

提供GNN模型定义、训练、评估和可视化功能。
"""

from src.model.gap_gnn import GapPredictionGNN
from src.model.training import train_epoch, evaluate, compute_loss
from src.model.visualization import plot_loss_curves, plot_prediction_scatter
from src.model.evaluation import calculate_metrics, metrics_to_dataframe
from src.model.scatter_plot import plot_actual_vs_predicted

__all__ = [
    "GapPredictionGNN",
    "train_epoch",
    "evaluate",
    "compute_loss",
    "plot_loss_curves",
    "plot_prediction_scatter",
    "calculate_metrics",
    "metrics_to_dataframe",
    "plot_actual_vs_predicted",
]
