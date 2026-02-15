"""
Merge QAC dataset with gap prediction results.

This script merges the base QAC dataset (with CID and CAS information)
with the gap calculation/prediction results, using SMILES as the primary key.
"""

import sys
from pathlib import Path

import pandas as pd

# Project path configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
QAC_DIR = DATA_DIR / "qac"
MODEL_DIR = PROJECT_ROOT / "models"

# Input files
BASE_CSV = QAC_DIR / "dataset_qac_c20_with_cid_with_cas.csv"
GAP_CSV = MODEL_DIR / "qac_c20_gap.csv"

# Output file
OUTPUT_CSV = DATA_DIR / "qac_results.csv"


def load_datasets():
    """Load the base dataset and gap results dataset."""
    print("Loading datasets...")
    base_df = pd.read_csv(BASE_CSV)
    gap_df = pd.read_csv(GAP_CSV)
    print(f"  Base dataset: {len(base_df)} rows")
    print(f"  Gap results: {len(gap_df)} rows")
    return base_df, gap_df


def merge_datasets(base_df, gap_df):
    """
    Merge datasets using SMILES as the primary key.
    
    Args:
        base_df: Base QAC dataset with CID and CAS information
        gap_df: Gap calculation/prediction results
        
    Returns:
        Merged dataframe with gap and predicted_gap as last columns
    """
    print("Merging datasets on SMILES...")
    
    # Extract only SMILES, gap, and predicted_gap from gap_df
    gap_subset = gap_df[['SMILES', 'gap', 'predicted_gap']].copy()
    
    # Merge using left join to keep all rows from base dataset
    merged_df = base_df.merge(
        gap_subset,
        on='SMILES',
        how='left'
    )
    
    # Report merge statistics
    matched = merged_df['predicted_gap'].notna().sum()
    print(f"  Matched rows: {matched}")
    print(f"  Unmatched rows: {len(merged_df) - matched}")
    
    return merged_df


def save_results(df, output_path):
    """Save merged dataset to CSV file."""
    print(f"Saving results to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows with {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")


def main():
    """Main function to merge QAC datasets."""
    print("=" * 60)
    print("QAC Dataset Merge Script")
    print("=" * 60)
    
    # Load datasets
    base_df, gap_df = load_datasets()
    
    # Merge datasets
    merged_df = merge_datasets(base_df, gap_df)
    
    # Save results
    save_results(merged_df, OUTPUT_CSV)
    
    print("=" * 60)
    print("Merge completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
