import argparse
import sys
from pathlib import Path

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT_PATH = DATA_DIR / "qac_results.csv"
DEFAULT_OUTPUT_PATH = DATA_DIR / "qac_plot.png"

REQUIRED_COLUMNS = {"SMILES", "xlogp", "gap", "predicted_gap"}


def has_any_cas_value(results_df: pd.DataFrame) -> pd.Series:
    cas_columns = [column for column in results_df.columns if column.endswith("_CAS")]
    if not cas_columns:
        raise ValueError("CSV missing CAS columns matching '*_CAS'.")

    cas_values = results_df[cas_columns].fillna("").astype(str)
    return cas_values.apply(lambda row: row.str.strip().ne("").any(), axis=1)


def configure_font() -> None:
    """Use Arial when available, otherwise fall back to a bundled font."""
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Arial", "Microsoft YaHei", "SimHei", "DejaVu Sans"):
        if font_name in installed_fonts:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            break

    plt.rcParams["axes.unicode_minus"] = False


def load_results(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    results_df = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(results_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV missing required columns: {missing}")

    gap_values = pd.to_numeric(results_df["gap"], errors="coerce")
    predicted_gap_values = pd.to_numeric(results_df["predicted_gap"], errors="coerce")
    results_df["gap_final"] = gap_values.combine_first(predicted_gap_values)
    results_df["xlogp"] = pd.to_numeric(results_df["xlogp"], errors="coerce")
    results_df = results_df.dropna(subset=["xlogp", "gap_final"]).copy()

    if results_df.empty:
        raise ValueError("No valid rows after converting xlogp/gap values to numbers.")

    return results_df


def build_plot(results_df: pd.DataFrame, output_path: Path, show: bool = False) -> None:
    candidate_condition = (results_df["xlogp"] <= 5) & has_any_cas_value(results_df)
    top3_gap_indices = (
        results_df.loc[candidate_condition]
        .nsmallest(3, "gap_final")
        .index
    )
    special_condition = results_df.index.isin(top3_gap_indices)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        results_df.loc[~special_condition, "xlogp"],
        results_df.loc[~special_condition, "gap_final"],
        c="#3C6AFF",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
        marker="o",
    )

    ax.scatter(
        results_df.loc[special_condition, "xlogp"],
        results_df.loc[special_condition, "gap_final"],
        c="#FF1D1D",
        alpha=0.9,
        s=120,
        edgecolors="black",
        linewidth=0.5,
        marker="*",
    )

    ax.set_xlabel("XlogP", fontsize=16)
    ax.set_ylabel("HOMO-LUMO Gap (eV)", fontsize=16)
    ax.set_title("QAC : XlogP vs Gap", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.axvline(x=5, color="#8A8A8A", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)

    print(f"Plot saved: {output_path}")
    print(f"Total samples: {len(results_df)}")
    print(f"Special condition (xlogP<=5, has *_CAS, Top 3 smallest Gap): {special_condition.sum()}")
    print(f"Others: {(~special_condition).sum()}")

    print("\n" + "=" * 70)
    print("Special Compounds (xlogP<=5, has *_CAS, Top 3 smallest Gap):")
    print("=" * 70)
    special_df = results_df.loc[
        special_condition,
        ["SMILES", "xlogp", "gap_final"],
    ].reset_index(drop=True)

    for idx, row in special_df.iterrows():
        print(f"{idx + 1}. SMILES: {row['SMILES']}")
        print(f"   xlogP: {row['xlogp']:.2f}, Gap: {row['gap_final']:.2f} eV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot QAC xlogP vs HOMO-LUMO gap.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--show", action="store_true", help="Display the plot window after saving.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_font()

    try:
        results_df = load_results(args.input)
        build_plot(results_df, args.output, show=args.show)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
