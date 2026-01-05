"""
数据完整性验证器

使用MD5哈希校验验证数据文件完整性。
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple


def compute_file_hash(file_path: str) -> str:
    """
    计算文件的MD5哈希值。
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: MD5哈希值（十六进制字符串）
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_checksum_path(data_path: str) -> Path:
    """
    获取校验和文件路径。
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        Path: 校验和文件路径 (.md5.json)
    """
    data_path = Path(data_path)
    return data_path.with_suffix(data_path.suffix + ".md5.json")


def save_checksum(
    data_path: str,
    metadata: Optional[Dict] = None
) -> str:
    """
    计算并保存数据文件的校验和。
    
    Args:
        data_path: 数据文件路径
        metadata: 额外的元数据（如生成参数、版本等）
        
    Returns:
        str: 校验和文件路径
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    file_hash = compute_file_hash(str(data_path))
    file_size = data_path.stat().st_size
    
    checksum_data = {
        "file_name": data_path.name,
        "md5": file_hash,
        "size_bytes": file_size,
        "created_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    
    checksum_path = get_checksum_path(str(data_path))
    with open(checksum_path, "w", encoding="utf-8") as f:
        json.dump(checksum_data, f, indent=2, ensure_ascii=False)
    
    return str(checksum_path)


def verify_checksum(data_path: str) -> Tuple[bool, str]:
    """
    验证数据文件的完整性。
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        Tuple[bool, str]: (是否验证通过, 验证信息)
    """
    data_path = Path(data_path)
    checksum_path = get_checksum_path(str(data_path))
    
    if not data_path.exists():
        return False, f"Data file not found: {data_path}"
    
    if not checksum_path.exists():
        return False, f"Checksum file not found: {checksum_path}"
    
    try:
        with open(checksum_path, "r", encoding="utf-8") as f:
            checksum_data = json.load(f)
    except json.JSONDecodeError:
        return False, "Checksum file is corrupted"
    
    expected_hash = checksum_data.get("md5")
    expected_size = checksum_data.get("size_bytes")
    
    if not expected_hash:
        return False, "Checksum file missing MD5 hash"
    
    # 验证文件大小
    actual_size = data_path.stat().st_size
    if expected_size and actual_size != expected_size:
        return False, f"File size mismatch: expected {expected_size}, got {actual_size}"
    
    # 验证MD5哈希
    actual_hash = compute_file_hash(str(data_path))
    if actual_hash != expected_hash:
        return False, f"MD5 mismatch: expected {expected_hash}, got {actual_hash}"
    
    created_at = checksum_data.get("created_at", "unknown")
    return True, f"Verified OK (created: {created_at})"


def check_data_integrity(
    data_path: str,
    verbose: bool = True
) -> bool:
    """
    检查数据文件完整性，如果验证通过则建议不重新生成。
    
    Args:
        data_path: 数据文件路径
        verbose: 是否打印验证信息
        
    Returns:
        bool: True表示数据完整，建议跳过重新生成
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        if verbose:
            print(f"[INFO] Data file not found: {data_path}")
        return False
    
    checksum_path = get_checksum_path(str(data_path))
    if not checksum_path.exists():
        if verbose:
            print(f"[INFO] No checksum file for: {data_path}")
            print("       Cannot verify integrity. Recommend regeneration.")
        return False
    
    is_valid, message = verify_checksum(str(data_path))
    
    if is_valid:
        if verbose:
            print(f"[OK] Data integrity verified: {data_path}")
            print(f"     {message}")
            print("     Existing data is valid. Regeneration not recommended.")
        return True
    else:
        if verbose:
            print(f"[WARNING] Data integrity check failed: {data_path}")
            print(f"          {message}")
            print("          Recommend regeneration.")
        return False


def get_checksum_metadata(data_path: str) -> Optional[Dict]:
    """
    获取校验和文件中的元数据。
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        dict: 元数据，如果不存在则返回None
    """
    checksum_path = get_checksum_path(data_path)
    
    if not checksum_path.exists():
        return None
    
    try:
        with open(checksum_path, "r", encoding="utf-8") as f:
            checksum_data = json.load(f)
        return checksum_data
    except (json.JSONDecodeError, IOError):
        return None
