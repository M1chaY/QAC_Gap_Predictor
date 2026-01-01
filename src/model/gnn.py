import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path


class GapPredictionGNN(torch.nn.Module):
    """
    用于预测HOMO-LUMO Gap的图神经网络
     
    轻量级架构（适合小数据集）：2层GAT + 简单MLP
    """
    
    def __init__(self, num_node_features, hidden_channels=80, num_global_features=3, num_heads=4):
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


def compute_loss(model, loader, device):
    """计算数据集的平均损失（用于绘图）"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = F.l1_loss(out.squeeze(), data.y)
            total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


def plot_loss_curves(train_losses, test_losses, save_path=None, title="Training and Test Loss Curves", figsize=(10, 6), dpi=600):
    """
    绘制训练集和测试集的损失曲线
    
    Args:
        train_losses: 训练集损失列表 (每个 epoch 的平均损失)
        test_losses: 测试集损失列表 (每个 epoch 的平均损失)
        save_path: 保存路径 (如果为 None, 则显示图像)
        title: 图表标题
        figsize: 图形尺寸
        dpi: 图形分辨率
    """
    # 设置字体
    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制损失曲线
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2, alpha=0.8)
    
    # 标注最小值
    min_train_idx = np.argmin(train_losses)
    min_test_idx = np.argmin(test_losses)
    
    ax.plot(min_train_idx + 1, train_losses[min_train_idx], 'bo', markersize=8)
    ax.plot(min_test_idx + 1, test_losses[min_test_idx], 'ro', markersize=8)
    
    ax.text(min_train_idx + 1, train_losses[min_train_idx], 
            f'  Min: {train_losses[min_train_idx]:.4f}',
            fontsize=14, verticalalignment='bottom')
    ax.text(min_test_idx + 1, test_losses[min_test_idx], 
            f'  Min: {test_losses[min_test_idx]:.4f}',
            fontsize=14, verticalalignment='top')
    
    # 设置标签和标题
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss (MAE)', fontsize=20)
    ax.set_title(title, fontsize=24, pad=10)
    
    # 设置图例和网格
    ax.legend(frameon=False, fontsize=20, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 设置刻度样式
    ax.tick_params(axis='both', which='both', length=5, width=2, colors='black', labelsize=16)
    
    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # 保存图形
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, transparent=False)
        print(f"\n✓ 损失曲线已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)