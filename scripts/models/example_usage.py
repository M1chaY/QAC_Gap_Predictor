"""
使用预处理数据集进行GNN训练的示例

演示如何使用input_graph_preparation.py生成的数据集
"""

import sys
import warnings
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

# 忽略torch-scatter警告（性能优化提示，不影响功能）
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.graph_dataset_loader import PreparedGraphDataset
from src.model.gnn import GapPredictionGNN, train_epoch, evaluate


def main():
    # 设置路径
    dataset_path = Path("data/qm9_prepared.joblib")  # 修改为你的数据集路径
    
    # 检查文件是否存在
    if not dataset_path.exists():
        print(f"❌ 数据集文件不存在: {dataset_path}")
        print("\n请先运行以下命令生成数据集:")
        print("python scripts/input_preparation/input_graph_preparation.py --input data/input.csv --output data/prepared_dataset.joblib")
        return
    
    # 加载数据集
    print("=" * 60)
    print("加载预处理数据集")
    print("=" * 60)
    
    dataset = PreparedGraphDataset(dataset_path)
    
    print(f"数据集信息:")
    print(f"  - 样本数量: {len(dataset)}")
    print(f"  - 节点特征维度: {dataset.num_node_features}")
    print(f"  - 边特征维度: {dataset.num_edge_features}")
    print(f"  - 全局特征维度: {dataset.num_global_features}")
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 验证集: {len(val_dataset)} 样本")
    
    # 创建DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoader配置:")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - 训练批次: {len(train_loader)}")
    print(f"  - 验证批次: {len(val_loader)}")
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    model = GapPredictionGNN(
        num_node_features=dataset.num_node_features,
        hidden_channels=80,
        num_global_features=dataset.num_global_features,
        num_heads=4
    ).to(device)
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练演示（仅1个epoch）
    print("\n" + "=" * 60)
    print("开始训练演示")
    print("=" * 60)
    
    num_epochs = 5
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # 验证
        val_r2, val_mae, val_rmse, _, _ = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val R²: {val_r2:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 训练演示完成！")
    print("=" * 60)
    
    # 展示如何访问元数据
    print("\n元数据访问示例:")
    print(f"  - SMILES列名: {dataset.info['smiles_col']}")
    print(f"  - 目标值列名: {dataset.info['target_col']}")
    print(f"  - 前5个SMILES:")
    for i, smiles in enumerate(dataset.metadata[dataset.info['smiles_col']].head(5)):
        print(f"    {i+1}. {smiles}")


if __name__ == "__main__":
    main()
