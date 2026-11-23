"""Parts 3 and 4: decode true missing values, keep structural "none", check keys.

Reads data/raw/AmesHousing.txt (never modified), writes data/processed/ames_clean.csv plus
data/processed/neighborhoods.csv, and markdown/CSV tables in reports/results/ for the report.
Every change is a rule (a condition on the data), not a hand-picked row, and every change is logged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import RESULTS, md_table, read_raw, write_result
from validate import ABSENT, NONE_CODE

# Columns where the dictionary defines "NA" as "none", not "unknown".
NA_MEANS_NONE: list[str] = [c for c, code in NONE_CODE.items() if code == "NA"]
# What "none" means in each structural column, quoted from the dictionary.
NONE_MEANING = {
    "Alley": "No alley access", "Bsmt Qual": "No Basement", "Bsmt Cond": "No Basement",
    "Bsmt Exposure": "No Basement", "BsmtFin Type 1": "No Basement",
    "BsmtFin Type 2": "No Basement", "Fireplace Qu": "No Fireplace", "Garage Type": "No Garage",
    "Garage Finish": "No Garage", "Garage Qual": "No Garage", "Garage Cond": "No Garage",
    "Pool QC": "No Pool", "Fence": "No Fence", "Misc Feature": "None",
    "Mas Vnr Type": "None (no masonry veneer)",
}

# Codes in the data that differ from the dictionary. Reported by validate.py on the raw file;
# here each one is mapped, explicitly, to the dictionary code it denotes, and the map is logged.
CODE_FIXES = {
    "MS Zoning": {"C (all)": "C", "I (all)": "I", "A (agr)": "A"},
    "Neighborhood": {"NAmes": "Names"},          # North Ames; the dictionary spells it "Names"
    "Bldg Type": {"Duplex": "Duplx", "Twnhs": "TwnhsI", "2fmCon": "2FmCon"},
    "Exterior 2nd": {"CmentBd": "CemntBd", "Wd Shng": "WdShing", "Brk Cmn": "BrkComm"},
}

# Neighborhood lookup (Part 4): code -> full name, from the dictionary.
NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale",
    "BrkSide": "Brookside", "ClearCr": "Clear Creek", "CollgCr": "College Creek",
    "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert", "Greens": "Greens",
    "GrnHill": "Green Hills", "IDOTRR": "Iowa DOT and Rail Road", "Landmrk": "Landmark",
    "MeadowV": "Meadow Village", "Mitchel": "Mitchell", "Names": "North Ames",
    "NoRidge": "Northridge", "NPkVill": "Northpark Villa", "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames", "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State University", "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook",
    "Timber": "Timberland", "Veenker": "Veenker",
}

# Numeric columns that use 0 for "none" (Part 3): which structural column says so, and the argument.
ZERO_ARGS = {
    "Mas Vnr Area": ("Mas Vnr Type", "True zero: no veneer has 0 sq ft. The 'none' itself is in Mas Vnr Type."),
    "Total Bsmt SF": ("Bsmt Qual", "True zero: 0 sq ft exactly when Bsmt Qual says No Basement (rule checked)."),
    "BsmtFin SF 1": ("BsmtFin Type 1", "True zero: unfinished or absent basement has 0 finished sq ft."),
    "BsmtFin SF 2": ("BsmtFin Type 2", "True zero, same as BsmtFin SF 1."),
    "Bsmt Unf SF": ("Bsmt Qual", "True zero: fully finished or absent basement."),
    "Bsmt Full Bath": ("Bsmt Qual", "True zero count; for no-basement houses it is also structural."),
    "Bsmt Half Bath": ("Bsmt Qual", "True zero count, as above."),
    "2nd Flr SF": ("House Style", "True zero: 1-story houses have no second floor."),
    "Low Qual Fin SF": ("-", "True zero: no low-quality finished area."),
    "Full Bath": ("-", "True zero count (above grade only; such houses have basement baths)."),
    "Half Bath": ("-", "True zero count."),
    "Bedroom AbvGr": ("-", "True zero: the dictionary excludes basement bedrooms from this count."),
    "Kitchen AbvGr": ("-", "Plausible true zero (above grade only), but Kitchen Qual is still rated: flagged, kept."),
    "Fireplaces": ("Fireplace Qu", "True zero: exactly when Fireplace Qu says No Fireplace (rule checked)."),
    "Garage Cars": ("Garage Type", "True zero: no garage holds 0 cars (Garage Type = No Garage)."),
    "Garage Area": ("Garage Type", "True zero: 0 sq ft exactly when there is no garage (rule checked)."),
    "Wood Deck SF": ("-", "True zero: no deck. No separate 'none' column exists."),
    "Open Porch SF": ("-", "True zero: no porch of this type."),
    "Enclosed Porch": ("-", "True zero: no porch of this type."),
    "3Ssn Porch": ("-", "True zero: no porch of this type."),
    "Screen Porch": ("-", "True zero: no porch of this type."),
    "Pool Area": ("Pool QC", "True zero: exactly when Pool QC says No Pool (rule checked)."),
    "Misc Val": ("Misc Feature", "Mostly structural (no feature). 3 rows have a feature with $0 value: kept as recorded."),
}


def load_raw(path: str) -> pd.DataFrame:
    """Load with keep_default_na=False so that "NA" and "None" survive as strings."""
    return read_raw(path)


# ---------------------------------------------------------------- Part 3 audit tables
def sentinel_audit(path: str, df: pd.DataFrame) -> None:
    """Compare the pandas default load with the literal load and write the Part 3 tables."""
    default = pd.read_csv(path, sep="\t")
    rows = []
    for c in df.columns:
        lost = default[c].isna()
        if not lost.any():
            continue
        tokens = df.loc[lost, c].fillna("<empty>").value_counts().to_dict()
        rows.append(dict(column=c, default_NaN=int(lost.sum()),
                         **{"as 'NA'": tokens.get("NA", 0), "as 'None'": tokens.get("None", 0),
                            "as empty": tokens.get("<empty>", 0)},
                         dictionary_none_code=NONE_CODE.get(c, "-"),
                         dictionary_meaning=NONE_MEANING.get(c, "no none-code: empty = unknown")))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "sentinels.csv", index=False)
    write_result("sentinels", md_table(t))

    other = sorted({tok for c in df.columns for tok in
                    df.loc[default[c].isna(), c].dropna().unique() if tok != "NA"})
    write_result("other_na_tokens", ", ".join(f"`{t}`" for t in other) or "none")

    z = []
    for c, (flag, arg) in ZERO_ARGS.items():
        z.append(dict(column=c, zeros=int((df[c] == 0).sum()), none_column=flag, argument=arg))
    write_result("zeros", md_table(pd.DataFrame(z)))


# ---------------------------------------------------------------- Part 3 decode
CHANGES: list[dict] = []


def log(step: str, column: str, mask, what: str, why: str) -> None:
    n = int(np.asarray(mask).sum())
    if n:
        CHANGES.append(dict(step=step, column=column, rows=n, what=what, why=why))


def decode_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Turn only true missing values into NaN. Return a new DataFrame."""
    df = df.copy()
    text = df.select_dtypes(exclude="number").columns.drop("PID")

    # 1. Whitespace: 'WD ' in Sale Type. The code is WD; the space is a formatting artefact.
    for c in text:
        m = df[c].notna() & (df[c] != df[c].str.strip())
        log("strip", c, m, "removed surrounding whitespace", "dictionary code has no trailing space")
        df[c] = df[c].str.strip()

    # 2. Codes that differ from the dictionary: explicit, logged map (reported by validate on raw).
    for c, fixes in CODE_FIXES.items():
        for old, new in fixes.items():
            m = df[c] == old
            log("code", c, m, f"'{old}' -> '{new}'", "dictionary code for the same category")
        df[c] = df[c].replace(fixes)

    # 3. Contradictions that make a recorded "none" unknown: a garage exists (Garage Type set,
    #    area > 0) but Qual/Cond say "No Garage". The "NA" cannot mean none here -> NaN.
    has_garage = df["Garage Type"].notna() & (df["Garage Type"] != "NA")
    for c in ["Garage Finish", "Garage Qual", "Garage Cond"]:
        m = has_garage & (df[c] == "NA")
        log("contradiction", c, m, "'NA' (no garage) -> NaN", "Garage Type says a garage exists: value unknown")
        df.loc[m, c] = np.nan

    # 4. Structural "none" kept as an explicit category, spelled ABSENT so no reader drops it.
    for c, code in NONE_CODE.items():
        m = df[c] == code
        log("structural", c, m, f"'{code}' -> '{ABSENT}'", f"dictionary: {NONE_MEANING[c]}")
        df.loc[m, c] = ABSENT

    # 5. Structural zeros that were left blank: no basement -> 0 basement bathrooms.
    no_bsmt = df["Bsmt Qual"] == ABSENT
    for c in ["Bsmt Full Bath", "Bsmt Half Bath", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF"]:
        m = no_bsmt & df[c].isna()
        log("structural", c, m, "NaN -> 0", "Bsmt Qual = No Basement, so the count/area is 0")
        df.loc[m, c] = 0

    # 6. Impossible or contradictory values become unknown (we cannot tell which field is wrong).
    m = df["Garage Yr Blt"] > 2010
    log("impossible", "Garage Yr Blt", m, "value after the study period -> NaN",
        "range rule; 2207 is probably 2007 but that is a guess")
    df.loc[m, "Garage Yr Blt"] = np.nan

    grace = np.where(df["Sale Condition"] == "Partial", 1, 0)
    m = (df["Year Remod/Add"] < df["Year Built"]) | (df["Year Remod/Add"] > df["Yr Sold"] + grace)
    log("contradiction", "Year Remod/Add", m, "-> NaN", "before construction or after the sale")
    df.loc[m, "Year Remod/Add"] = np.nan

    none_type = df["Mas Vnr Type"] == ABSENT
    zero_area = df["Mas Vnr Area"] == 0
    m = df["Mas Vnr Type"].notna() & df["Mas Vnr Area"].notna() & (none_type != zero_area)
    for c in ["Mas Vnr Type", "Mas Vnr Area"]:
        log("contradiction", c, m, "-> NaN", "type and area disagree; either could be the error")
        df.loc[m, c] = np.nan

    # Empty fields were already read as NaN (true unknowns). Garage Yr Blt is NaN for the 157
    # houses without a garage: structural, recoverable from Garage Type == ABSENT, never imputed.
    return df


def missing_summary(df: pd.DataFrame) -> None:
    no_garage = df["Garage Type"] == ABSENT
    rows = []
    for c in df.columns[df.isna().any()]:
        n = int(df[c].isna().sum())
        structural = int((df[c].isna() & no_garage).sum()) if c == "Garage Yr Blt" else 0
        rows.append(dict(column=c, NaN=n, structural=structural, true_unknown=n - structural,
                         share=round(n / len(df), 4)))
    write_result("missing_after", md_table(pd.DataFrame(rows), floatfmt="{:.4f}"))
    ch = pd.DataFrame(CHANGES)
    ch.to_csv(RESULTS / "changes.csv", index=False)
    write_result("changes", md_table(ch))


# ---------------------------------------------------------------- Part 4
def check_keys(df: pd.DataFrame, raw_path: str, out_dir: Path = Path("data/processed")) -> None:
    """Assert key uniqueness; print exact and key duplicate counts; run the many_to_one join."""
    default = pd.read_csv(raw_path, sep="\t")      # how PID looks with the pandas default
    keys = pd.DataFrame([
        dict(key="Order", unique=df["Order"].is_unique, duplicates=int(df["Order"].duplicated().sum()),
             identifies="a row of this file (observation number 1..2930)"),
        dict(key="PID", unique=df["PID"].is_unique, duplicates=int(df["PID"].duplicated().sum()),
             identifies="a parcel (a property), not a sale"),
        dict(key="PID + Yr Sold + Mo Sold", unique=not df.duplicated(["PID", "Yr Sold", "Mo Sold"]).any(),
             duplicates=int(df.duplicated(["PID", "Yr Sold", "Mo Sold"]).sum()),
             identifies="a sale: one parcel sold in one month"),
    ])
    dups = pd.DataFrame([
        dict(check="exact duplicate rows", count=int(df.duplicated().sum())),
        dict(check="duplicates ignoring Order", count=int(df.drop(columns="Order").duplicated().sum())),
        dict(check="duplicates ignoring Order and PID",
             count=int(df.drop(columns=["Order", "PID"]).duplicated().sum())),
        dict(check="PIDs with a leading 0 (lost by the default load)", count=int(df["PID"].str.startswith("0").sum())),
        dict(check="PID length != 10 after default load (int)",
             count=int((default["PID"].astype(str).str.len() != 10).sum())),
    ])
    assert df["Order"].is_unique and df["PID"].is_unique, "key is not unique"

    lookup = pd.DataFrame(sorted(NEIGHBORHOODS.items()), columns=["Neighborhood", "Neighborhood Name"])
    lookup.to_csv(out_dir / "neighborhoods.csv", index=False)
    joined = pd.merge(df, lookup, on="Neighborhood", how="left", validate="many_to_one", indicator=True)
    unmatched = int((joined["_merge"] != "both").sum())
    assert len(joined) == len(df), "join changed the row count"
    assert unmatched == 0, f"{unmatched} sales have a neighborhood code missing from the lookup"

    print(f"Keys: Order unique={df['Order'].is_unique}, PID unique={df['PID'].is_unique}; "
          f"exact duplicates={dups['count'][0]}; join rows {len(df)} -> {len(joined)}, unmatched={unmatched}")
    write_result("keys", md_table(keys))
    write_result("duplicates", md_table(dups))
    write_result("join", md_table(pd.DataFrame([dict(
        rows_before=len(df), rows_after=len(joined), lookup_rows=len(lookup),
        unmatched=unmatched, neighborhoods_in_data=df["Neighborhood"].nunique())])))


if __name__ == "__main__":
    raw_path, out_path = sys.argv[1], sys.argv[2]
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = load_raw(raw_path)
    print(f"Raw: {raw.shape[0]} rows, {raw.shape[1]} columns")
    sentinel_audit(raw_path, raw)
    df = decode_missing(raw)
    missing_summary(df)
    check_keys(df, raw_path, Path(out_path).parent)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {df.shape[0]} rows, {df.shape[1]} columns, {len(CHANGES)} logged changes")
