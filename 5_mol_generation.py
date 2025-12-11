import csv
from pathlib import Path
from typing import Iterable, Optional, Tuple, List
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers


def _iter_smiles(csv_path: Path) -> Iterable[Tuple[str, str, str]]:
    """Yield (index, num_c, smiles) rows from a CSV with header."""
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["Index"], row["Num_c"], row["SMILES"].strip()


def _build_3d_mol(smiles: str) -> Optional[Chem.Mol]:
    # 从SMILES生成3D分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 加入氢原子
    mol = Chem.AddHs(mol)

    # 生成3D构象
    params = rdDistGeom.ETKDGv3()
    params.randomSeed = 0xF00D
    embed_result = rdDistGeom.EmbedMolecule(mol, params)
    if embed_result == -1:
        # If ETKDG fails, try basic embedding
        embed_result = rdDistGeom.EmbedMolecule(mol)
        if embed_result == -1:
            print(f"Warning: Could not generate 3D coordinates for SMILES: {smiles}")
            return None

    # Try MMFF optimization first
    try:
        mmff_result = rdForceFieldHelpers.MMFFOptimizeMolecule(mol)
        if mmff_result == 1:  # 1 indicates failure, 0 indicates success
            # If MMFF fails, try UFF as fallback
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
    except ValueError:
        # If MMFF completely fails, try UFF
        try:
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
        except ValueError:
            print(f"Warning: Could not optimize molecule for SMILES: {smiles}")
            # Return the molecule anyway - it has 3D coordinates even if not optimized
            pass

    return mol


def _write_single_mol(mol: Chem.Mol, output_path: Path) -> bool:
    """Write a single molecule to an MDL MOL file."""
    try:
        Chem.MolToMolFile(mol, str(output_path))
        return True
    except Exception as e:
        print(f"Warning: failed to write MOL to {output_path}: {e}")
        return False


def _zero_pad_index(raw_idx: str, width: int, fallback_seq: int) -> str:
    """Return zero-padded index like 0001 using the given width. If raw_idx is not an int, use fallback_seq."""
    try:
        n = int(raw_idx)
    except (TypeError, ValueError):
        n = fallback_seq
    return str(n).zfill(width)


def main() -> None:
    """Entry point: convert CSV SMILES to 3D and write EACH molecule into its own .mol file."""
    root = Path(__file__).resolve().parent
    csv_path = root / "data" / "r4n_c20_validated_pubchempy.csv"
    output_dir = root / "data" / "mol"

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating 3D molecules from SMILES (RDKit)...")
    print(f"Input:  {csv_path}")
    print(f"Output dir (MOL files): {output_dir}")

    rows = list(_iter_smiles(csv_path))
    print(f"Total rows in CSV: {len(rows)}")

    # First, build molecules and collect the successful ones
    successful: List[Tuple[str, str, str, Chem.Mol]] = []
    failed_count = 0

    for idx, num_c, smi in rows:
        mol = _build_3d_mol(smi)
        if mol is None:
            failed_count += 1
            continue
        successful.append((idx, num_c, smi, mol))

    total_out = len(successful)
    width = len(str(total_out)) if total_out > 0 else 1
    print(f"Will write {total_out} molecules to individual MOL files (index width={width}). Failed builds: {failed_count}")

    written = 0
    for seq, (idx, num_c, smi, mol) in enumerate(successful, start=1):
        padded_idx = _zero_pad_index(idx, width, seq)

        # Set name and properties (only the title is preserved in MOL; properties are for traceability while in-memory)
        try:
            mol.SetProp("_Name", f"{padded_idx}_{smi}")
        except Exception:
            pass
        mol.SetProp("Index", padded_idx)
        mol.SetProp("Num_c", str(num_c))
        mol.SetProp("SMILES", smi)

        out_path = output_dir / f"{padded_idx}_{smi}.mol"
        if _write_single_mol(mol, out_path):
            written += 1

    print("Done.")
    print(f"Successfully written: {written}")
    print(f"Failed: {failed_count}")


if __name__ == "__main__":
    main()
