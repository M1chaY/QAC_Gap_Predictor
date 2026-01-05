"""
PubChem验证流水线

提供完整的PubChem验证流水线，支持断点续传。
"""

from pathlib import Path

from src.data.r4n.step1_cid import step1_validate_cid
from src.data.r4n.step2_properties import step2_add_properties
from src.data.r4n.step3_halide_cas import step3_query_halide_cas


def run_full_validation_pipeline(
    input_csv: str,
    get_properties: bool = True,
    get_halide_cas: bool = True,
    verbose: bool = True
) -> str:
    """
    运行完整的PubChem验证流水线。
    
    每步独立保存，支持断点续传。如果某步的输出文件已存在且校验通过，
    则跳过该步骤。
    
    Args:
        input_csv: 基础化合物CSV路径
        get_properties: 是否获取额外属性（Step 2）
        get_halide_cas: 是否查询卤化盐CAS（Step 3）
        verbose: 是否打印进度
        
    Returns:
        str: 最终输出文件路径
    """
    from src.io.integrity import check_data_integrity
    
    input_path = Path(input_csv)
    stem = input_path.stem
    
    # Step 1: CID验证
    cid_csv = str(input_path.with_name(stem + "_with_cid.csv"))
    if check_data_integrity(cid_csv, verbose=False):
        if verbose:
            print(f"[Step 1] CID validation already done: {cid_csv}")
        current_csv = cid_csv
    else:
        current_csv = step1_validate_cid(input_csv, cid_csv, verbose)
    
    # Step 2: 获取属性
    if get_properties:
        props_csv = str(input_path.with_name(stem + "_with_props.csv"))
        if check_data_integrity(props_csv, verbose=False):
            if verbose:
                print(f"[Step 2] Properties already fetched: {props_csv}")
            current_csv = props_csv
        else:
            current_csv = step2_add_properties(current_csv, props_csv, verbose)
    
    # Step 3: 查询卤化盐CAS
    if get_halide_cas:
        cas_csv = str(input_path.with_name(stem + "_with_cas.csv"))
        if check_data_integrity(cas_csv, verbose=False):
            if verbose:
                print(f"[Step 3] Halide CAS already queried: {cas_csv}")
            current_csv = cas_csv
        else:
            current_csv = step3_query_halide_cas(current_csv, cas_csv, verbose)
    
    if verbose:
        print(f"\n[OK] Pipeline complete. Final output: {current_csv}")
    
    return current_csv
