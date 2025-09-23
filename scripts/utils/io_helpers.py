import pandas as pd
from pathlib import Path
from scripts.utils.constants import apply_canonical_columns

def read_csv_canonical(path: Path | str, **kwargs) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, comment="#", **kwargs)
    return apply_canonical_columns(df)

def read_parquet_canonical(path: Path | str, **kwargs) -> pd.DataFrame:
    df = pd.read_parquet(path, **kwargs)
    return apply_canonical_columns(df)

def load_overlay(path: Path | str) -> pd.DataFrame:
    path = Path(path)  # ensure Path object
    if path.suffix == ".csv":
        return read_csv_canonical(path)
    elif path.suffix == ".parquet":
        return read_parquet_canonical(path)
    else:
        raise ValueError(f"Unsupported overlay format: {path}")

