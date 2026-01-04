"""
文件加载器

提供CSV和Excel文件加载功能。
"""

from pathlib import Path
from typing import Optional

import pandas as pd


def load_input_file(
    file_path: Path, 
    sheet_name: Optional[str] = None
) -> pd.DataFrame:
    """
    加载输入文件（CSV或Excel）。
    
    Args:
        file_path: 输入文件路径
        sheet_name: Excel工作表名（仅对Excel文件有效）
        
    Returns:
        pd.DataFrame: 加载的数据
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不支持或Excel未指定工作表
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == ".csv":
        print(f"Reading CSV file: {file_path}")
        return pd.read_csv(file_path)
    
    elif suffix in [".xlsx", ".xls"]:
        if sheet_name is None:
            raise ValueError("Excel file requires sheet_name parameter")
        print(f"Reading Excel file: {file_path}, sheet: {sheet_name}")
        return pd.read_excel(file_path, sheet_name=sheet_name)
    
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}, only .csv, .xlsx, .xls supported"
        )
