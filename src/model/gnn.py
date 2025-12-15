import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


class GapPredictionGNN(torch.nn.Module):
    """
    用于预测HOMO-LUMO Gap的图神经网络
     
    轻量级架构（适合小数据集）：2层GAT + 简单MLP
    """
    
    def __init__(self, num_node_features, hidden_channels=80, num_global_features=4, num_heads=4):
        super().__init__()
        
        # 确保hidden_channels能被num_heads整除
        assert hidden_channels % num_heads == 0, "hidden_channels must be divisible by num_heads"
        
        # 2层GAT（注意力机制比GCN更有效）
        self.conv1 = GATConv(num_node_features, hidden_channels // num_heads, heads=num_heads, dropout=0.2)
        self.conv2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=0.2)
        
        # 批归一化（只在MLP中使用）
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # 全局特征
        self.num_global_features = num_global_features
        
        # 轻量MLP（mean + max池化）
        mlp_input = hidden_channels * 2 + num_global_features
        self.lin1 = nn.Linear(mlp_input, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, 1)
        
        self.dropout = nn.Dropout(0.18)  # 轻微提高dropout
    
    def forward(self, data):
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
        
        # MLP（在第一层后使用BatchNorm）
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.lin2(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.lin3(x)
        
        return x
    

def train_epoch(model, loader, optimizer, device):
    """训练一个epoch"""
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


def evaluate(model, loader, device):
    """评估模型"""
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