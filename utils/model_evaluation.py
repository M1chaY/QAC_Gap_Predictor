import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


def _calculate_metrics(
        y_true: pd.Series,
        y_pred: pd.Series
) -> tuple[float, float, float, float]:
    """
    Calculate and return actual vs pred fig for data_dopants metrics.
    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    Returns
    -------
    r2 : float
        R-squared score.
    mae : float
        Mean Absolute Error.
    mape : float
        Mean Absolute Percentage Error.
    rmse : float
        Root Mean Squared Error.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, mape, rmse


def metrics_to_df(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        model_name
):
    """
    Convert metrics to DataFrame for better visualization.
    Parameters
    ----------
    y_train : array-like
        Actual values for the training set.
    y_train_pred : array-like
        Predicted values for the training set.
    y_test : array-like
        Actual values for the test set.
    y_test_pred : array-like
        Predicted values for the test set.
    model_name : str
        Name of the model.
    Returns
    -------
    pd.DataFrame([metrics]) : DataFrame
        Containing the calculated metrics.
    """

    r2_train, mae_train, mape_train, rmse_train = _calculate_metrics(y_train, y_train_pred)
    r2_test, mae_test, mape_test, rmse_test = _calculate_metrics(y_test, y_test_pred)
    metrics = {
        'model': model_name,
        'R2_train': round(r2_train, 3),
        'MAE_train': round(mae_train, 2),
        'MAPE_train': round(mape_train, 2),
        'RMSE_train': round(rmse_train, 2),
        'R2_test': round(r2_test, 3),
        'MAE_test': round(mae_test, 2),
        'MAPE_test': round(mape_test, 2),
        'RMSE_test': round(rmse_test, 2)}
    return pd.DataFrame([metrics])


def scat_avp(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    save_path: str,
    axis_min: float = 0,
    axis_max: float = 220,
    model_name: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
    dpi: int = 600
) -> None:
    """绘制实际值vs预测值对比图

    Parameters:
    -----------
    y_train, y_train_pred : array-like
        训练集实际值和预测值
    y_test, y_test_pred : array-like
        测试集实际值和预测值
    save_path : str
        图片保存路径
    axis_min, axis_max : float
        坐标轴范围
    model_name : str
        模型名称
    title : str, optional
        图表标题
    figsize : Tuple[int, int]
        图形尺寸
    dpi : int
        图形分辨率
    """
    # 设置默认标题
    if title is None:
        title = f'{model_name}'

    # 创建图形
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 绘制散点图
    ax.scatter(y_train, y_train_pred, color='blue', label='Train', s=50, alpha=0.3)
    ax.scatter(y_test, y_test_pred, color='red', label='Test', s=50, alpha=0.3)
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', lw=2, label='Perfect Prediction')

    # 设置坐标轴范围和刻度
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    major_ticks = np.arange(axis_min, axis_max, 0.5)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)

    # 设置标签和标题
    ax.set_xlabel('Actual Values', fontsize=20)
    ax.set_ylabel('Predicted Values', fontsize=20)
    ax.set_title(title, fontsize=24, pad=10)

    # 设置图例和刻度样式
    ax.legend(frameon=False, fontsize=20, loc='upper left')
    ax.tick_params(axis='both', which='both', length=5, width=2, colors='black', labelsize=16)

    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')

    # 保存图形
    plt.tight_layout()
    fig.savefig(save_path, transparent=False)
    plt.close(fig)