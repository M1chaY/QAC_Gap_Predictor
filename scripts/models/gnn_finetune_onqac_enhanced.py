"""
QAC GNN微调脚本

加载QM9预训练模型，使用QAC数据集进行微调。

策略：
1. 读取预训练模型和配置
2. 加载QAC数据集，选取指定数量样本用于训练
3. 五折交叉验证训练，以平均MAE作为评估指标
4. 剩余样本作为测试集
5. 可变学习率（ReduceLROnPlateau）
6. 保存训练/测试loss用于后续绘图
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from src.model.enhanced_finetune_utils import (
    evaluate_test_set_with_state,
    aggregate_fold_histories,
    build_finetune_info,
    save_finetune_outputs,
)
from src import (
    PreparedGraphDataset,
    GapPredictionGNN,
    evaluate,
    set_seed
)

# 忽略torch-scatter警告
warnings.filterwarnings("ignore", message=".*torch-scatter.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
PARAMS_DIR = MODEL_DIR / "params"

# 节点特征中3D坐标列索引: [one-hot(4), x, y, z, ...]
COORD_START_IDX = 4
COORD_DIM = 3


def load_pretrained_checkpoint(model_path: Path, device: torch.device):
    """
    加载预训练模型检查点。
    
    Args:
        model_path: 预训练模型路径
        device: 计算设备
        
    Returns:
        dict: 检查点字典
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    
    print(f"Loading pretrained model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"  Best epoch: {checkpoint['epoch']}")
    print(f"  Val MAE: {checkpoint['val_mae']:.4f}")
    
    return checkpoint


def create_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """
    从检查点创建并初始化模型。
    
    Args:
        checkpoint: 模型检查点
        device: 计算设备
        
    Returns:
        GapPredictionGNN: 初始化后的模型
    """
    model_config = checkpoint['model_config']
    dataset_info = checkpoint['dataset_info']
    
    model = GapPredictionGNN(
        num_node_features=dataset_info['num_node_features'],
        gat_dims=model_config['gat_dims'],
        mlp_dims=model_config['mlp_dims'],
        num_global_features=dataset_info['num_global_features'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model


def resolve_weight_decay(
    checkpoint: dict,
    model_config: dict,
    default_weight_decay: float = 1e-4,
) -> tuple[float, str]:
    """
    解析微调使用的weight_decay，优先使用预训练时优化器中的真实值。

    Args:
        checkpoint: 预训练检查点
        model_config: 模型配置字典
        default_weight_decay: 默认值

    Returns:
        tuple: (weight_decay, 来源描述)
    """
    optimizer_state = checkpoint.get('optimizer_state_dict', None)
    if optimizer_state and optimizer_state.get('param_groups'):
        wd = optimizer_state['param_groups'][0].get('weight_decay', None)
        if wd is not None:
            return float(wd), "checkpoint.optimizer_state_dict.param_groups[0].weight_decay"

    if 'weight_decay' in model_config:
        return float(model_config['weight_decay']), "checkpoint.model_config.weight_decay"

    return float(default_weight_decay), f"default({default_weight_decay})"


def augment_graph_batch(
    data,
    coord_noise_std: float,
    node_mask_ratio: float,
):
    """
    对批图数据执行增强：坐标高斯噪声 + 节点特征随机Mask。

    Args:
        data: PyG Batch数据
        coord_noise_std: 坐标噪声标准差
        node_mask_ratio: 节点特征mask比例

    Returns:
        Batch: 增强后的批数据
    """
    if not hasattr(data, 'x') or data.x is None:
        return data

    data_aug = data.clone()
    data_aug.x = data_aug.x.clone()
    x = data_aug.x

    # 随机mask节点特征（与坐标噪声解耦：不mask xyz列）
    if node_mask_ratio > 0:
        feat_dim = x.size(1)
        coord_end_idx = min(COORD_START_IDX + COORD_DIM, feat_dim)
        mask = torch.rand_like(x) < node_mask_ratio
        if feat_dim > COORD_START_IDX:
            mask[:, COORD_START_IDX:coord_end_idx] = False
        x = x.masked_fill(mask, 0.0)

    # 对坐标列添加高斯噪声
    if coord_noise_std > 0 and x.size(1) > COORD_START_IDX:
        coord_end_idx = min(COORD_START_IDX + COORD_DIM, x.size(1))
        noise = torch.randn_like(x[:, COORD_START_IDX:coord_end_idx]) * coord_noise_std
        x[:, COORD_START_IDX:coord_end_idx] = x[:, COORD_START_IDX:coord_end_idx] + noise

    data_aug.x = x
    return data_aug


def train_epoch_with_augmentation(
    model,
    loader,
    optimizer,
    device,
    coord_noise_std: float,
    node_mask_ratio: float,
) -> float:
    """训练一个epoch，训练阶段对输入图执行数据增强。"""
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        data = augment_graph_batch(data, coord_noise_std, node_mask_ratio)

        optimizer.zero_grad()
        out = model(data)
        loss = F.l1_loss(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def configure_trainable_layers(model: GapPredictionGNN, strategy: str = "C") -> dict:
    """
    配置可训练层。

    策略说明：
    - A: 全部层参与微调（不冻结）
    - B: 冻结全部GAT层，仅训练MLP头部
    - C: 冻结前面GAT层，仅训练最后一层GAT和全部MLP头部

    Args:
        model: GNN模型
        strategy: 微调策略

    Returns:
        dict: 训练参数统计信息
    """
    normalized = strategy.strip().upper()

    # 为兼容旧配置，保留别名写法
    if strategy == "last_gat_only":
        normalized = "C"
    elif strategy == "all_trainable":
        normalized = "A"
    elif strategy == "head_only":
        normalized = "B"

    # 默认先全部冻结，后续按策略定向解冻
    for param in model.parameters():
        param.requires_grad = False

    if normalized == "A":
        # A: 全部可训练
        for param in model.parameters():
            param.requires_grad = True
        strategy_desc = "all_trainable"
    elif normalized == "B":
        # B: 仅训练MLP头部
        model.mlp_layers.requires_grad_(True)
        model.mlp_bns.requires_grad_(True)
        model.output_layer.requires_grad_(True)
        strategy_desc = "head_only"
    elif normalized == "C":
        # C: 最后一层GAT + MLP头部
        model.gat_layers[-1].requires_grad_(True)
        model.gat_bns[-1].requires_grad_(True)
        model.mlp_layers.requires_grad_(True)
        model.mlp_bns.requires_grad_(True)
        model.output_layer.requires_grad_(True)
        strategy_desc = "last_gat_plus_head"
    else:
        raise ValueError(
            f"Unsupported fine-tune strategy: {strategy}. Use A, B, or C."
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'strategy': normalized,
        'strategy_desc': strategy_desc,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0,
    }


def run_single_fold(
    fold_idx: int,
    n_folds: int,
    fold_train: list,
    fold_val: list,
    checkpoint: dict,
    device: torch.device,
    freeze_strategy: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    patience: int,
    coord_noise_std: float,
    node_mask_ratio: float,
    min_delta: float,
) -> dict:
    """执行单折严格CV训练并返回该折结果。"""
    train_loader = DataLoader(fold_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(fold_val, batch_size=batch_size, shuffle=False)

    print("-" * 60)
    print(f"Fold {fold_idx}/{n_folds}: Train {len(fold_train)}, Val {len(fold_val)}")

    # 每一折都从同一预训练权重重新开始，确保折间独立
    model = create_model_from_checkpoint(checkpoint, device)
    trainable_stats = configure_trainable_layers(model, strategy=freeze_strategy)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters after applying freeze strategy")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(
        f"Trainable parameters: {trainable_stats['trainable_params']:,} / "
        f"{trainable_stats['total_params']:,} "
        f"({trainable_stats['trainable_ratio']:.2%})"
    )

    epoch_train_losses = []
    epoch_val_losses = []
    best_mae = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        # 训练阶段启用增强；验证阶段不做增强
        train_loss = train_epoch_with_augmentation(
            model,
            train_loader,
            optimizer,
            device,
            coord_noise_std=coord_noise_std,
            node_mask_ratio=node_mask_ratio,
        )
        _, val_mae, _, _, _ = evaluate(model, val_loader, device)

        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_mae)

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']

        is_new_best = val_mae < best_mae - min_delta
        if epoch % 10 == 0 or epoch == 1 or is_new_best:
            best_marker = " *NEW BEST*" if is_new_best else ""
            print(
                f"Fold {fold_idx:02d} | Epoch {epoch:04d}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val MAE={val_mae:.4f} | "
                f"LR={current_lr:.2e}{best_marker}"
            )

        if is_new_best:
            best_mae = val_mae
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping in fold {fold_idx} at epoch {epoch}")
                print(f"  No improvement > {min_delta} for {patience} epochs")
                break

    if best_state is None:
        # 兜底逻辑：理论上几乎不会触发，确保best_state始终可用
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = len(epoch_val_losses)
        best_mae = float(np.min(epoch_val_losses))

    print(f"Fold {fold_idx} done: Best Val MAE={best_mae:.4f} at epoch {best_epoch}")
    return {
        'best_state': best_state,
        'best_epoch': best_epoch,
        'best_mae': best_mae,
        'epoch_train_losses': np.array(epoch_train_losses, dtype=float),
        'epoch_val_losses': np.array(epoch_val_losses, dtype=float),
        'trainable_stats': trainable_stats,
    }


def build_split_indices(
    total_size: int,
    num_samples: int,
    seed: int,
    legacy_split: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """构建训练/测试索引，支持旧版与新版随机划分方式。"""
    if legacy_split:
        # 旧版行为：RandomState(seed).permutation
        all_indices = np.random.RandomState(seed).permutation(total_size)
    else:
        # 新版行为：default_rng(seed).permutation
        all_indices = np.random.default_rng(seed).permutation(total_size)
    return all_indices[:num_samples], all_indices[num_samples:]


def run_final_retrain_on_full_train(
    checkpoint: dict,
    train_data: list,
    device: torch.device,
    freeze_strategy: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    retrain_epochs: int,
    coord_noise_std: float,
    node_mask_ratio: float,
) -> tuple[dict, dict]:
    """在全部训练池样本上重训最终模型（用于与旧流程可比）。"""
    model = create_model_from_checkpoint(checkpoint, device)
    trainable_stats = configure_trainable_layers(model, strategy=freeze_strategy)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters after applying freeze strategy")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    print("\n" + "=" * 60)
    print("Final retrain on full train pool")
    print("=" * 60)
    print(f"Train samples: {len(train_data)}")
    print(f"Retrain epochs: {retrain_epochs}")

    for epoch in range(1, retrain_epochs + 1):
        train_loss = train_epoch_with_augmentation(
            model,
            train_loader,
            optimizer,
            device,
            coord_noise_std=coord_noise_std,
            node_mask_ratio=node_mask_ratio,
        )
        if epoch % 10 == 0 or epoch == 1 or epoch == retrain_epochs:
            print(f"Final Epoch {epoch:04d}/{retrain_epochs}: Train Loss={train_loss:.4f}")

    final_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return final_state, trainable_stats


def finetune_on_qac(
    pretrained_path: Path,
    dataset_path: Path,
    num_samples: int = 50,
    n_folds: int = 5,
    num_epochs: int = 200,
    patience: int = 30,
    batch_size: int = 4,
    lr: float = 1e-4,
    freeze_strategy: str = "C",
    legacy_split: bool = True,
    coord_noise_std: float = 0.01,
    node_mask_ratio: float = 0.1,
    seed: int = 42
) -> bool:
    """
    使用QAC数据集进行严格五折交叉验证微调。
    
    Args:
        pretrained_path: 预训练模型路径
        dataset_path: QAC数据集路径
        num_samples: 用于训练的样本数量
        n_folds: 交叉验证折数
        num_epochs: 最大训练轮数
        patience: 早停耐心值
        batch_size: 批次大小
        lr: 初始学习率
        freeze_strategy: 微调冻结策略（A/B/C）
            A=全部可训练, B=仅MLP头部, C=最后一层GAT+MLP头部
        legacy_split: 是否使用旧版随机划分（RandomState）
        coord_noise_std: 坐标高斯噪声标准差
        node_mask_ratio: 节点特征随机mask比例
        seed: 随机种子
        
    Returns:
        bool: 是否成功
    """
    try:
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载预训练检查点
        print("\n" + "=" * 60)
        print("Loading pretrained model")
        print("=" * 60)

        checkpoint = load_pretrained_checkpoint(pretrained_path, device)
        model_config = checkpoint['model_config']
        weight_decay, weight_decay_source = resolve_weight_decay(checkpoint, model_config)

        print(f"\nModel config:")
        print(f"  GAT dims: {model_config['gat_dims']}")
        print(f"  MLP dims: {model_config['mlp_dims']}")
        print(f"  Weight decay: {weight_decay:.6g} ({weight_decay_source})")

        # 加载QAC数据集
        print("\n" + "=" * 60)
        print("Loading QAC dataset")
        print("=" * 60)

        dataset = PreparedGraphDataset(dataset_path)
        print(f"\nDataset info:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Node features: {dataset.num_node_features}")
        print(f"  Global features: {dataset.num_global_features}")

        # 划分训练集和测试集
        if num_samples > len(dataset):
            print(f"Warning: requested {num_samples}, only {len(dataset)} available")
            num_samples = len(dataset)

        train_indices, test_indices = build_split_indices(
            total_size=len(dataset),
            num_samples=num_samples,
            seed=seed,
            legacy_split=legacy_split,
        )
        train_data = [dataset[i] for i in train_indices]
        test_data = [dataset[i] for i in test_indices]

        print(f"\nData split:")
        print(f"  Train (for CV): {len(train_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        print(f"\nDevice: {device}")

        print("\n" + "=" * 60)
        print(f"{n_folds}-Fold Cross Validation Finetuning")
        print("=" * 60)
        print(f"Learning rate: {lr}")
        print(f"Freeze strategy: {freeze_strategy} (A/B/C)")
        print(f"Legacy split: {legacy_split}")
        print(f"Coord noise std: {coord_noise_std}")
        print(f"Node mask ratio: {node_mask_ratio}")
        print(f"Epochs: {num_epochs}, Patience: {patience}\n")

        if len(train_data) < n_folds:
            raise ValueError(
                f"Train samples ({len(train_data)}) must be >= n_folds ({n_folds})"
            )

        # 原始五折逻辑：单一模型，每个epoch依次使用5折训练，再对5折验证集评估
        fold_splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(train_data))
        fold_loaders = []
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits, start=1):
            fold_train = [train_data[i] for i in train_idx]
            fold_val = [train_data[i] for i in val_idx]
            train_loader = DataLoader(fold_train, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(fold_val, batch_size=batch_size, shuffle=False)
            fold_loaders.append((train_loader, val_loader))
            print(f"Fold {fold_idx}: Train {len(fold_train)}, Val {len(fold_val)}")

        model = create_model_from_checkpoint(checkpoint, device)
        trainable_stats = configure_trainable_layers(model, strategy=freeze_strategy)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters after applying freeze strategy")

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7
        )

        print(f"\nModel parameters: {model.count_parameters():,}")
        print(
            f"Trainable parameters: {trainable_stats['trainable_params']:,} / "
            f"{trainable_stats['total_params']:,} "
            f"({trainable_stats['trainable_ratio']:.2%})"
        )

        min_delta = 1e-4
        best_avg_mae = float('inf')
        best_epoch = 0
        best_state = None
        patience_counter = 0

        epoch_train_losses = []
        epoch_val_losses = []
        fold_val_losses = [[] for _ in range(n_folds)]

        print("\nSequential 5-Fold training started...\n")
        for epoch in range(1, num_epochs + 1):
            epoch_fold_train = []
            epoch_fold_val = []

            # 每个epoch内遍历5折进行训练
            for train_loader, _ in fold_loaders:
                train_loss = train_epoch_with_augmentation(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    coord_noise_std=coord_noise_std,
                    node_mask_ratio=node_mask_ratio,
                )
                epoch_fold_train.append(train_loss)

            # 每个epoch后对5折验证集评估
            for fold_idx, (_, val_loader) in enumerate(fold_loaders):
                _, val_mae, _, _, _ = evaluate(model, val_loader, device)
                epoch_fold_val.append(val_mae)
                fold_val_losses[fold_idx].append(val_mae)

            avg_train_loss = float(np.mean(epoch_fold_train))
            avg_val_mae = float(np.mean(epoch_fold_val))
            epoch_train_losses.append(avg_train_loss)
            epoch_val_losses.append(avg_val_mae)

            scheduler.step(avg_val_mae)
            current_lr = optimizer.param_groups[0]['lr']

            is_new_best = avg_val_mae < best_avg_mae - min_delta
            if epoch % 10 == 0 or epoch == 1 or is_new_best:
                fold_maes = ", ".join([f"{mae:.4f}" for mae in epoch_fold_val])
                best_marker = " *NEW BEST*" if is_new_best else ""
                print(
                    f"Epoch {epoch:04d}/{num_epochs}: "
                    f"Train Loss={avg_train_loss:.4f}, "
                    f"Avg Val MAE={avg_val_mae:.4f} | "
                    f"LR={current_lr:.2e}{best_marker}"
                )
                print(f"         Fold MAEs: [{fold_maes}]")

            if is_new_best:
                best_avg_mae = avg_val_mae
                best_epoch = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(f"  No improvement > {min_delta} for {patience} epochs")
                    break

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = len(epoch_val_losses)
            best_avg_mae = float(np.min(epoch_val_losses))

        fold_best_maes = [float(np.min(losses)) for losses in fold_val_losses]
        fold_best_epochs = [int(np.argmin(losses)) + 1 for losses in fold_val_losses]
        best_fold_idx = int(np.argmin(fold_best_maes))

        print("\n" + "=" * 60)
        print("Sequential CV summary")
        print("=" * 60)
        for i, (mae, ep) in enumerate(zip(fold_best_maes, fold_best_epochs), start=1):
            print(f"Fold {i}: Best Val MAE={mae:.4f} at epoch {ep}")
        print(f"Best Avg Val MAE: {best_avg_mae:.4f} at epoch {best_epoch}")

        print("\n" + "=" * 60)
        print("Test set evaluation")
        print("=" * 60)
        test_r2, test_mae, test_rmse, has_test_set = evaluate_test_set_with_state(
            create_model_fn=create_model_from_checkpoint,
            evaluate_fn=evaluate,
            checkpoint=checkpoint,
            best_state=best_state,
            test_data=test_data,
            batch_size=batch_size,
            device=device,
        )

        finetuned_path = MODEL_DIR / "qac_finetuned_enhanced.pt"
        history_path = PARAMS_DIR / "qac_finetune_history_enhanced.npz"
        finetune_info = build_finetune_info(
            num_samples=num_samples,
            n_folds=n_folds,
            has_test_set=has_test_set,
            freeze_strategy=freeze_strategy,
            trainable_stats=trainable_stats,
            lr=lr,
            weight_decay=weight_decay,
            weight_decay_source=weight_decay_source,
            coord_noise_std=coord_noise_std,
            node_mask_ratio=node_mask_ratio,
            pretrained_path=pretrained_path,
        )
        finetune_info["legacy_split"] = legacy_split
        finetune_info["final_retrain_epochs"] = 0
        finetune_info["final_model_source"] = "sequential_cv_best_epoch"
        finetune_info["cv_selected_best_fold"] = best_fold_idx + 1

        save_finetune_outputs(
            params_dir=PARAMS_DIR,
            finetuned_path=finetuned_path,
            history_path=history_path,
            best_state=best_state,
            model_config=model_config,
            best_epoch=best_epoch,
            best_avg_mae=best_avg_mae,
            test_r2=test_r2,
            test_mae=test_mae,
            test_rmse=test_rmse,
            dataset=dataset,
            finetune_info=finetune_info,
            epoch_train_losses=np.array(epoch_train_losses, dtype=float),
            epoch_val_losses=np.array(epoch_val_losses, dtype=float),
            fold_val_losses=np.array(fold_val_losses, dtype=float),
            fold_best_maes=fold_best_maes,
            fold_best_epochs=fold_best_epochs,
            best_fold_idx=best_fold_idx,
            has_test_set=has_test_set,
        )

        return True

    except Exception as e:
        print(f"\nFinetuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数。"""
    print("=" * 60)
    print("QAC GNN Finetuning Script (5-Fold CV)")
    print("=" * 60)
    
    pretrained_path = MODEL_DIR / "qm9_pretrained.pt"
    dataset_path = DATA_DIR / "qac_prepared.joblib"
    
    # 检查文件存在性
    if not pretrained_path.exists():
        print(f"Error: Pretrained model not found: {pretrained_path}")
        sys.exit(1)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # 执行微调（使用100条数据进行5折交叉验证，24条作为测试集）
    success = finetune_on_qac(
        pretrained_path=pretrained_path,
        dataset_path=dataset_path,
        num_samples=100,
        n_folds=5,
        num_epochs=1000,
        patience=50,
        batch_size=8,
        lr=0.001531,
        freeze_strategy="C",
        legacy_split=True,
        coord_noise_std=0.01,
        node_mask_ratio=0.1,
        seed=42
    )
    
    if success:
        print("\n" + "=" * 60)
        print("Finetuning completed successfully")
        print("=" * 60)
    else:
        print("\nFinetuning failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
