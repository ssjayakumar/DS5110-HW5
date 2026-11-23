"""Shared helpers: one way to read each file, and small markdown writers for the report."""
from pathlib import Path

import pandas as pd

RESULTS = Path("reports/results")
FIGURES = Path("reports/figures")

# PID is a nominal code with leading zeros (data dictionary, References section),
# so it is always read as text.
READ_KW = dict(keep_default_na=False, na_values=[""], dtype={"PID": str})


def read_raw(path: str) -> pd.DataFrame:
    """Raw file: tab-separated. Only empty fields become NaN; "NA"/"None" stay strings."""
    return pd.read_csv(path, sep="\t", **READ_KW)


def read_clean(path: str) -> pd.DataFrame:
    """Cleaned file: comma-separated. Same rule, so "Absent" and codes survive as written."""
    return pd.read_csv(path, **READ_KW)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}", index: bool = False) -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    if index:
        df = df.reset_index()

    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|")

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def write_result(name: str, text: str) -> None:
    """Write a markdown fragment to reports/results/<name>.md (used by src/report.py)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.md").write_text(text.strip() + "\n", encoding="utf-8")
