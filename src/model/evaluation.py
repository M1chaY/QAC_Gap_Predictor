"""
模型评估指标计算

提供回归模型的评估指标计算功能。
"""

import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


def calculate_metrics(y_true, y_pred) -> tuple:
    """
    计算回归模型评估指标。
    
    Args:
        y_true: 实际值
        y_pred: 预测值
        
    Returns:
        tuple: (r2, mae, mape, rmse)
            - r2: R-squared
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error (%)
            - rmse: Root Mean Squared Error
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    rmse = root_mean_squared_error(y_true, y_pred)
    return r2, mae, mape, rmse


def metrics_to_dataframe(
    y_train,
    y_train_pred,
    y_test,
    y_test_pred,
    model_name: str
) -> pd.DataFrame:
    """
    将训练集和测试集的评估指标转换为DataFrame。
    
    Args:
        y_train: 训练集实际值
        y_train_pred: 训练集预测值
        y_test: 测试集实际值
        y_test_pred: 测试集预测值
        model_name: 模型名称
        
    Returns:
        pd.DataFrame: 包含所有评估指标的DataFrame
    """
    r2_train, mae_train, mape_train, rmse_train = calculate_metrics(
        y_train, y_train_pred
    )
    r2_test, mae_test, mape_test, rmse_test = calculate_metrics(
        y_test, y_test_pred
    )
    
    metrics = {
        'model': model_name,
        'R2_train': round(r2_train, 3),
        'MAE_train': round(mae_train, 2),
        'MAPE_train': round(mape_train, 2),
        'RMSE_train': round(rmse_train, 2),
        'R2_test': round(r2_test, 3),
        'MAE_test': round(mae_test, 2),
        'MAPE_test': round(mape_test, 2),
        'RMSE_test': round(rmse_test, 2),
    }
    
    return pd.DataFrame([metrics])
