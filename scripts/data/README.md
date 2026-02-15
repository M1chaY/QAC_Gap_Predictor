# Data Processing Scripts

This directory contains scripts for data processing and merging operations.

## Scripts

### merge_qac_results.py

Merges the QAC base dataset with gap calculation/prediction results.

**Input Files:**

- `data/qac/dataset_qac_c20_with_cid_with_cas.csv` - Base dataset with CID and CAS
- `models/qac_c20_gap.csv` - Gap calculation and prediction results

**Output File:**

- `data/qac_results.csv` - Merged dataset with gap values

**Usage:**

```bash
python scripts/data/merge_qac_results.py
```

**Merge Logic:**

- Uses SMILES as the primary key for matching
- Performs left join to preserve all rows from the base dataset
- Appends `gap` and `predicted_gap` columns at the end
