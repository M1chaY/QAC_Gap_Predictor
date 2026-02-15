# Dataset Preparation Scripts

This directory contains scripts for preparing and processing datasets
for the QAC Gap Predictor project.

## Scripts Overview

### dataset_qm9_workflow.py

Interactive workflow script for QM9 dataset preparation.

Features:

- Extract QM9 data from PyTorch Geometric
- Preprocess dataset with molecular feature computation
- Full workflow combining extraction and preprocessing

Usage:

```bash
python scripts/dataset_preparation/dataset_qm9_workflow.py
```

### dataset_qac_workflow.py

Interactive workflow script for QAC cation dataset preparation.

Features:

- Generate QAC quaternary ammonium compounds (basic, no PubChem)
- Step-by-step PubChem validation pipeline:
  - Step 1: CID validation with MD5 checksum
  - Step 2: Property fetching (IUPAC name, molecular weight, etc.)
  - Step 3: Halide salt CAS number query
- Clean dataset by removing invalid entries
- Data integrity verification using MD5 checksums

Usage:

```bash
python scripts/dataset_preparation/dataset_qac_workflow.py
```

Programmatic usage for step-by-step validation:

```python
from src import (
    step1_validate_cid,
    step2_add_properties,
    step3_query_halide_cas,
    run_full_validation_pipeline,
)

# Run individual steps
step1_validate_cid("input.csv", "output_with_cid.csv")
step2_add_properties("output_with_cid.csv", "output_with_props.csv")
step3_query_halide_cas("output_with_props.csv", "output_with_cas.csv")

# Or run the full pipeline
run_full_validation_pipeline("input.csv", verbose=True)
```

### structure_qac_generation.py

Script for generating 3D molecular structure files (.mol) from SMILES.

Features:

- Read SMILES list from CSV file
- Generate 3D coordinates using RDKit (via `generate_mol_files`)
- Save molecular structures as MOL files

Usage:

```bash
python scripts/dataset_preparation/structure_qac_generation.py
```

## Dependencies

These scripts require the `src` module and the following packages:

- rdkit
- pandas
- torch-geometric (for QM9)

## Output Locations

- QM9 data: `data/qm9/`
- QAC data: `data/qac/`
- Molecular structures: `data/qac/mol_files/`
