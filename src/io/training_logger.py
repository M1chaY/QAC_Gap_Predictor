"""
训练日志记录工具

提供Optuna超参数搜索和模型训练的日志记录功能。
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Union, Optional


def setup_training_logger(
    log_dir: Union[str, Path],
    log_name: str = "training",
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置训练日志记录器。
    
    Args:
        log_dir: 日志输出目录
        log_name: 日志名称（用于文件名和logger名）
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_name}_{timestamp}.log"
    
    # 创建logger
    logger = logging.getLogger(f"{log_name}_{timestamp}")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_trial_start(logger: logging.Logger, trial_number: int, params: dict):
    """
    记录trial开始和超参数。
    
    Args:
        logger: 日志记录器
        trial_number: trial编号
        params: 超参数字典
    """
    logger.info("=" * 60)
    logger.info(f"TRIAL {trial_number} STARTED")
    logger.info("-" * 40)
    logger.info("Hyperparameters:")
    for key, value in params.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("-" * 40)


def log_epoch(
    logger: logging.Logger,
    trial_number: int,
    epoch: int,
    train_loss: float,
    val_mae: float,
    val_r2: float,
    lr: float,
    is_best: bool = False
):
    """
    记录单个epoch的训练结果。
    
    Args:
        logger: 日志记录器
        trial_number: trial编号
        epoch: epoch编号
        train_loss: 训练损失
        val_mae: 验证集MAE
        val_r2: 验证集R2
        lr: 当前学习率
        is_best: 是否为最佳epoch
    """
    best_marker = " [BEST]" if is_best else ""
    logger.info(
        f"Trial {trial_number:03d} | Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | "
        f"Val R2: {val_r2:.4f} | LR: {lr:.6f}{best_marker}"
    )


def log_trial_end(
    logger: logging.Logger,
    trial_number: int,
    best_val_mae: float,
    total_epochs: int,
    status: str = "COMPLETED"
):
    """
    记录trial结束。
    
    Args:
        logger: 日志记录器
        trial_number: trial编号
        best_val_mae: 最佳验证MAE
        total_epochs: 总训练epoch数
        status: 结束状态（COMPLETED/PRUNED/EARLY_STOPPED）
    """
    logger.info("-" * 40)
    logger.info(f"TRIAL {trial_number} {status}")
    logger.info(f"  Best Val MAE: {best_val_mae:.4f}")
    logger.info(f"  Total Epochs: {total_epochs}")
    logger.info("=" * 60)
    logger.info("")


def log_search_summary(
    logger: logging.Logger,
    best_params: dict,
    best_val_mae: float,
    test_metrics: dict,
    n_completed: int,
    n_pruned: int
):
    """
    记录搜索总结。
    
    Args:
        logger: 日志记录器
        best_params: 最佳超参数
        best_val_mae: 最佳验证MAE
        test_metrics: 测试集指标
        n_completed: 完成的trial数
        n_pruned: 被剪枝的trial数
    """
    logger.info("=" * 60)
    logger.info("HYPERPARAMETER SEARCH SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Completed Trials: {n_completed}")
    logger.info(f"Pruned Trials: {n_pruned}")
    logger.info("")
    logger.info("Best Hyperparameters:")
    for key, value in best_params.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("")
    logger.info(f"Best Validation MAE: {best_val_mae:.4f}")
    logger.info("")
    logger.info("Test Set Performance:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 60)
