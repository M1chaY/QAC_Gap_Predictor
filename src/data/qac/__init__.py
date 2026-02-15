"""
QAC数据集处理模块

提供季铵离子(QAC)化合物的生成、验证和清洗功能。
"""

from src.data.qac.smiles_builder import build_qac_smiles
from src.data.qac.alkyl_groups import generate_alkyl_groups
from src.data.qac.molecule_validator import validate_qac_molecule, get_canonical_smiles
from src.data.qac.pubchem_query import (
    validate_pubchem_compound,
    add_halide_to_smiles,
    get_cas_number,
    PUBCHEM_AVAILABLE,
)
from src.data.qac.generator import QACGenerator
from src.data.qac.cleaner import clean_qac_dataset
from src.data.qac.statistics import get_statistics, print_statistics, save_compounds_to_csv
from src.data.qac.step1_cid import step1_validate_cid
from src.data.qac.step2_properties import step2_add_properties
from src.data.qac.step3_halide_cas import step3_query_halide_cas
from src.data.qac.pubchem_pipeline import run_full_validation_pipeline
from src.data.qac.preprocessor import preprocess_qac_dataset

__all__ = [
    "build_qac_smiles",
    "generate_alkyl_groups",
    "validate_qac_molecule",
    "get_canonical_smiles",
    "validate_pubchem_compound",
    "add_halide_to_smiles",
    "get_cas_number",
    "PUBCHEM_AVAILABLE",
    "QACGenerator",
    "clean_qac_dataset",
    "get_statistics",
    "print_statistics",
    "save_compounds_to_csv",
    # 分步验证
    "step1_validate_cid",
    "step2_add_properties",
    "step3_query_halide_cas",
    "run_full_validation_pipeline",
    # 预处理
    "preprocess_qac_dataset",
]

