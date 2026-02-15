# src/config.py  或  src/paths.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
QM9_DIR = DATA_DIR / "qm9"
QAC_DIR = DATA_DIR / "qac"
