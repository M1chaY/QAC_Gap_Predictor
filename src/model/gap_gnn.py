"""
Gap预测GNN模型

用于预测HOMO-LUMO Gap的图神经网络模型定义。
支持每层独立配置维度，用于Optuna超参数搜索。
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

# 基础倍率
BASE_UNIT = 16


class GapPredictionGNN(torch.nn.Module):
    """
    用于预测HOMO-LUMO Gap的可配置图神经网络。
     
    支持每层独立配置维度：
    - GAT层：每层可独立设置维度（基础倍率的1-8倍）
    - MLP层：每层可独立设置维度（基础倍率的1-8倍）
    - Mean + Max全局池化
    
    Args:
        num_node_features: 节点特征维度
        gat_dims: GAT每层维度列表，如[64, 32]表示2层
        mlp_dims: MLP每层维度列表，如[128, 64, 32]表示3层（不含输出层）
        num_edge_features: 边特征维度（默认1，键级特征）
        num_global_features: 全局分子描述符数量（默认3）
        num_heads: GAT注意力头数（默认4）
        dropout: Dropout概率（默认0.2）
    """
    
    def __init__(
        self, 
        num_node_features: int,
        gat_dims: List[int],
        mlp_dims: List[int],
        num_edge_features: int = 1,
        num_global_features: int = 3, 
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        assert len(gat_dims) >= 1, "At least 1 GAT layer is required"
        assert len(mlp_dims) >= 1, "At least 1 MLP layer is required"
        
        # 验证所有GAT维度可被num_heads整除
        for i, dim in enumerate(gat_dims):
            assert dim % num_heads == 0, \
                f"GAT layer {i} dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.gat_dims = gat_dims
        self.mlp_dims = mlp_dims
        self.num_edge_features = num_edge_features
        self.num_heads = num_heads
        self.num_global_features = num_global_features
        self.dropout_rate = dropout
        
        # 构建GAT层
        self.gat_layers = nn.ModuleList()
        self.gat_bns = nn.ModuleList()
        
        # 第一层GAT：输入特征 -> gat_dims[0]，包含边特征
        self.gat_layers.append(GATConv(
            num_node_features, 
            gat_dims[0] // num_heads, 
            heads=num_heads, 
            edge_dim=num_edge_features,
            dropout=dropout
        ))
        self.gat_bns.append(nn.BatchNorm1d(gat_dims[0]))
        
        # 后续GAT层
        for i in range(1, len(gat_dims)):
            self.gat_layers.append(GATConv(
                gat_dims[i-1], 
                gat_dims[i] // num_heads, 
                heads=num_heads, 
                edge_dim=num_edge_features,
                dropout=dropout
            ))
            self.gat_bns.append(nn.BatchNorm1d(gat_dims[i]))
        
        # 构建MLP层
        self.mlp_layers = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        # MLP输入维度：最后一层GAT的mean + max池化 + 全局特征
        mlp_input = gat_dims[-1] * 2 + num_global_features
        
        # 第一层MLP
        self.mlp_layers.append(nn.Linear(mlp_input, mlp_dims[0]))
        self.mlp_bns.append(nn.BatchNorm1d(mlp_dims[0]))
        
        # 后续MLP层
        for i in range(1, len(mlp_dims)):
            self.mlp_layers.append(nn.Linear(mlp_dims[i-1], mlp_dims[i]))
            self.mlp_bns.append(nn.BatchNorm1d(mlp_dims[i]))
        
        # 输出层
        self.output_layer = nn.Linear(mlp_dims[-1], 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        前向传播。
        
        Args:
            data: PyG Data对象，需包含x, edge_index, edge_attr, batch, 可选u
            
        Returns:
            torch.Tensor: 预测的Gap值，形状为(batch_size, 1)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # GAT层前向传播（包含边特征）
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.gat_bns)):
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            if i < len(self.gat_layers) - 1:
                x = self.dropout(x)
        
        # 全局池化（mean + max）
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 拼接全局分子描述符
        if hasattr(data, 'u') and data.u is not None:
            x = torch.cat([x, data.u], dim=1)
        
        # MLP前向传播
        for layer, bn in zip(self.mlp_layers, self.mlp_bns):
            x = layer(x)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)
        
        # 输出层（无激活函数）
        x = self.output_layer(x)
        
        return x
    
    @classmethod
    def from_config(
        cls, 
        config: dict, 
        num_node_features: int, 
        num_global_features: int,
        num_edge_features: int = 1
    ):
        """
        从配置字典创建模型。
        
        Args:
            config: 包含模型超参数的字典
            num_node_features: 节点特征维度
            num_global_features: 全局特征维度
            num_edge_features: 边特征维度（默认1）
            
        Returns:
            GapPredictionGNN: 模型实例
        """
        return cls(
            num_node_features=num_node_features,
            gat_dims=config['gat_dims'],
            mlp_dims=config['mlp_dims'],
            num_edge_features=num_edge_features,
            num_global_features=num_global_features,
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.2)
        )
    
    def get_config(self) -> dict:
        """
        获取模型配置字典。
        
        Returns:
            dict: 模型超参数配置
        """
        return {
            'gat_dims': self.gat_dims,
            'mlp_dims': self.mlp_dims,
            'num_heads': self.num_heads,
            'dropout': self.dropout_rate
        }
    
    def count_parameters(self) -> int:
        """统计模型参数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
