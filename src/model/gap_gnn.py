"""
Gap预测GNN模型

用于预测HOMO-LUMO Gap的图神经网络模型定义。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class GapPredictionGNN(torch.nn.Module):
    """
    用于预测HOMO-LUMO Gap的图神经网络。
     
    轻量级架构（适合小数据集）：
    - 2层GAT（图注意力网络）
    - 简单MLP回归头
    - Mean + Max全局池化
    
    Args:
        num_node_features: 节点特征维度
        hidden_channels: 隐藏层维度（默认80）
        num_global_features: 全局分子描述符数量（默认3）
        num_heads: GAT注意力头数（默认4）
    """
    
    def __init__(
        self, 
        num_node_features: int, 
        hidden_channels: int = 80, 
        num_global_features: int = 3, 
        num_heads: int = 4
    ):
        super().__init__()
        
        assert hidden_channels % num_heads == 0, \
            "hidden_channels must be divisible by num_heads"
        
        # 2层GAT
        self.conv1 = GATConv(
            num_node_features, 
            hidden_channels // num_heads, 
            heads=num_heads, 
            dropout=0.2
        )
        self.conv2 = GATConv(
            hidden_channels, 
            hidden_channels // num_heads, 
            heads=num_heads, 
            dropout=0.2
        )
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # 全局特征
        self.num_global_features = num_global_features
        
        # MLP回归头（mean + max池化）
        mlp_input = hidden_channels * 2 + num_global_features
        self.lin1 = nn.Linear(mlp_input, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, 1)
        
        self.dropout = nn.Dropout(0.18)
    
    def forward(self, data):
        """
        前向传播。
        
        Args:
            data: PyG Data对象，需包含x, edge_index, batch, 可选u
            
        Returns:
            torch.Tensor: 预测的Gap值，形状为(batch_size, 1)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT层 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # GAT层 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # 全局池化（mean + max）
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 拼接全局分子描述符
        if hasattr(data, 'u'):
            x = torch.cat([x, data.u], dim=1)
        
        # MLP
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.lin2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.lin3(x)
        
        return x
