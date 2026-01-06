"""
数据集划分工具

提供数据集划分和随机种子设置功能。
"""

import numpy as np
import torch


def set_seed(seed: int):
    """
    设置全局随机种子以确保可复现性。
    
    Args:
        seed: 随机种子值
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    按比例划分数据集为训练集、验证集和测试集。
    
    Args:
        dataset: 数据集（支持索引访问）
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.15）
        seed: 随机种子（默认42）
        
    Returns:
        tuple: (训练集列表, 验证集列表, 测试集列表)
        
    Example:
        >>> train, val, test = split_dataset(dataset, 0.7, 0.15, seed=42)
        >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    """
    set_seed(seed)
    
    n = len(dataset)
    indices = np.random.permutation(n)
    
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    return train_dataset, val_dataset, test_dataset
