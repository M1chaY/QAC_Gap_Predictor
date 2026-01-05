"""
R4N数据集处理模块

提供季铵离子(R4N+)化合物的生成、验证和清洗功能。
"""

from src.data.r4n.smiles_builder import build_r4n_smiles
from src.data.r4n.alkyl_groups import generate_alkyl_groups
from src.data.r4n.molecule_validator import validate_r4n_molecule, get_canonical_smiles
from src.data.r4n.pubchem_query import (
    validate_pubchem_compound,
    add_halide_to_smiles,
    get_cas_number,
    PUBCHEM_AVAILABLE,
)
from src.data.r4n.generator import R4NGenerator
from src.data.r4n.cleaner import clean_r4n_dataset
from src.data.r4n.statistics import get_statistics, print_statistics, save_compounds_to_csv
from src.data.r4n.step1_cid import step1_validate_cid
from src.data.r4n.step2_properties import step2_add_properties
from src.data.r4n.step3_halide_cas import step3_query_halide_cas
from src.data.r4n.pubchem_pipeline import run_full_validation_pipeline
from src.data.r4n.preprocessor import preprocess_r4n_dataset

__all__ = [
    "build_r4n_smiles",
    "generate_alkyl_groups",
    "validate_r4n_molecule",
    "get_canonical_smiles",
    "validate_pubchem_compound",
    "add_halide_to_smiles",
    "get_cas_number",
    "PUBCHEM_AVAILABLE",
    "R4NGenerator",
    "clean_r4n_dataset",
    "get_statistics",
    "print_statistics",
    "save_compounds_to_csv",
    # 分步验证
    "step1_validate_cid",
    "step2_add_properties",
    "step3_query_halide_cas",
    "run_full_validation_pipeline",
    # 预处理
    "preprocess_r4n_dataset",
]

