"""
模型训练工具

提供GNN模型的训练和评估功能。
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def train_epoch(model, loader, optimizer, device) -> float:
    """
    训练一个epoch。
    
    Args:
        model: PyTorch模型
        loader: 数据加载器
        optimizer: 优化器
        device: 计算设备
        
    Returns:
        float: 平均损失
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = F.l1_loss(out.squeeze(), data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device) -> tuple:
    """
    评估模型性能。
    
    Args:
        model: PyTorch模型
        loader: 数据加载器
        device: 计算设备
        
    Returns:
        tuple: (r2, mae, rmse, predictions, targets)
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            predictions.extend(out.squeeze().cpu().numpy())
            targets.extend(data.y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)

    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)

    return r2, mae, rmse, predictions, targets


def compute_loss(model, loader, device) -> float:
    """
    计算数据集的平均损失。
    
    Args:
        model: PyTorch模型
        loader: 数据加载器
        device: 计算设备
        
    Returns:
        float: 平均损失（MAE）
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.l1_loss(out.squeeze(), data.y)
            total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)
