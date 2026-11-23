"""Part 2: validation as code. Raise once, listing every failure (notebook §1 pattern, plain pandas).

Usage:  python src/validate.py data/raw/AmesHousing.txt --raw
        python src/validate.py data/processed/ames_clean.csv
Exit code 1 when any rule fails. With --raw, the failure table is also saved to
reports/results/validation_raw.csv for the report. With --expect-fail the exit code is inverted
(0 when validation fails, 1 when it passes), so the Makefile needs no shell "!" and runs on Windows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import read_clean, read_raw

DICT = "data dictionary (De Cock 2011, DataDocumentation.txt)"
STUDY = "study period: sales 2006-2010 (dictionary abstract)"
PHYS = "physical limit"
AMES = "physical limit: Ames was founded in 1864, no building can be older"

# How the cleaned file spells the dictionary's "none" code (NA = no garage, None = no veneer).
# A word that no CSV reader turns into NaN by default (unlike "NA" and "None").
ABSENT = "Absent"


class ValidationError(Exception):
    def __init__(self, failures: list[dict]):
        self.failures = failures
        super().__init__("\n".join(f["message"] for f in failures))


# ---------------------------------------------------------------- schema
# 82 columns as named in the data file (the dictionary spells a few differently, e.g.
# "Exterior 1" for "Exterior 1st", "HeatingQC" for "Heating QC"; the file is the contract).
TEXT_COLUMNS = [
    "PID", "MS Zoning", "Street", "Alley", "Lot Shape", "Land Contour", "Utilities",
    "Lot Config", "Land Slope", "Neighborhood", "Condition 1", "Condition 2", "Bldg Type",
    "House Style", "Roof Style", "Roof Matl", "Exterior 1st", "Exterior 2nd", "Mas Vnr Type",
    "Exter Qual", "Exter Cond", "Foundation", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
    "BsmtFin Type 1", "BsmtFin Type 2", "Heating", "Heating QC", "Central Air", "Electrical",
    "Kitchen Qual", "Functional", "Fireplace Qu", "Garage Type", "Garage Finish",
    "Garage Qual", "Garage Cond", "Paved Drive", "Pool QC", "Fence", "Misc Feature",
    "Sale Type", "Sale Condition",
]
NUMERIC_COLUMNS = [
    "Order", "MS SubClass", "Lot Frontage", "Lot Area", "Overall Qual", "Overall Cond",
    "Year Built", "Year Remod/Add", "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2",
    "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF",
    "Gr Liv Area", "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath",
    "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces", "Garage Yr Blt",
    "Garage Cars", "Garage Area", "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
    "3Ssn Porch", "Screen Porch", "Pool Area", "Misc Val", "Mo Sold", "Yr Sold", "SalePrice",
]
EXPECTED_COLUMNS = set(TEXT_COLUMNS) | set(NUMERIC_COLUMNS)   # 44 + 38 = 82

# ---------------------------------------------------------------- range rules
# column: (low, high, source). Bounds are inclusive; NaN is not judged here.
INF = np.inf
RANGES = {
    "Order":          (1, 2930, "dictionary: observation number, SIZE 2930 observations"),
    "Yr Sold":        (2006, 2010, STUDY),
    "Mo Sold":        (1, 12, "physical limit: calendar months (dictionary: MM)"),
    "Year Built":     (1864, 2010, AMES + "; upper: " + STUDY),
    "Year Remod/Add": (1864, 2010, AMES + "; upper: " + STUDY),
    "Garage Yr Blt":  (1864, 2010, AMES + "; upper: " + STUDY),
    "Overall Qual":   (1, 10, DICT + ": rating 1 Very Poor to 10 Very Excellent"),
    "Overall Cond":   (1, 10, DICT + ": rating 1 Very Poor to 10 Very Excellent"),
    "Lot Area":       (1, INF, PHYS + ": a lot has positive area"),
    "Lot Frontage":   (1, INF, PHYS + ": feet of street, positive when recorded"),
    "1st Flr SF":     (1, INF, PHYS + ": every dwelling has a first floor"),
    "Gr Liv Area":    (1, INF, PHYS + ": a dwelling has living area"),
    "TotRms AbvGrd":  (1, INF, PHYS + ": a dwelling has at least one room"),
    "SalePrice":      (1, INF, PHYS + ": a sale has a positive price"),
    **{c: (0, INF, PHYS + ": areas cannot be negative") for c in [
        "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
        "2nd Flr SF", "Low Qual Fin SF", "Garage Area", "Wood Deck SF", "Open Porch SF",
        "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Pool Area"]},
    **{c: (0, INF, PHYS + ": counts cannot be negative") for c in [
        "Bsmt Full Bath", "Bsmt Half Bath", "Full Bath", "Half Bath", "Bedroom AbvGr",
        "Kitchen AbvGr", "Fireplaces", "Garage Cars"]},
    "Misc Val":       (0, INF, PHYS + ": a dollar value cannot be negative"),
}

# ---------------------------------------------------------------- allowed values
# Every code listed in the data dictionary, copied verbatim. "NA"/"None" are the
# dictionary's own "none" codes; ABSENT is how the cleaned file writes them.
QUAL5 = {"Ex", "Gd", "TA", "Fa", "Po"}
ALLOWED = {
    "MS SubClass": {20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190},
    "MS Zoning": {"A", "C", "FV", "I", "RH", "RL", "RP", "RM"},
    "Street": {"Grvl", "Pave"},
    "Alley": {"Grvl", "Pave", "NA"},
    "Lot Shape": {"Reg", "IR1", "IR2", "IR3"},
    "Land Contour": {"Lvl", "Bnk", "HLS", "Low"},
    "Utilities": {"AllPub", "NoSewr", "NoSeWa", "ELO"},
    "Lot Config": {"Inside", "Corner", "CulDSac", "FR2", "FR3"},
    "Land Slope": {"Gtl", "Mod", "Sev"},
    "Neighborhood": {
        "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards",
        "Gilbert", "Greens", "GrnHill", "IDOTRR", "Landmrk", "MeadowV", "Mitchel", "Names",
        "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW",
        "Somerst", "StoneBr", "Timber", "Veenker"},
    "Condition 1": {"Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"},
    "Condition 2": {"Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"},
    "Bldg Type": {"1Fam", "2FmCon", "Duplx", "TwnhsE", "TwnhsI"},
    "House Style": {"1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"},
    "Roof Style": {"Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"},
    "Roof Matl": {"ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"},
    "Exterior 1st": {
        "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc",
        "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"},
    "Exterior 2nd": {
        "AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc",
        "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing"},
    "Mas Vnr Type": {"BrkCmn", "BrkFace", "CBlock", "None", "Stone"},
    "Exter Qual": QUAL5,
    "Exter Cond": QUAL5,
    "Foundation": {"BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"},
    "Bsmt Qual": QUAL5 | {"NA"},
    "Bsmt Cond": QUAL5 | {"NA"},
    "Bsmt Exposure": {"Gd", "Av", "Mn", "No", "NA"},
    "BsmtFin Type 1": {"GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"},
    "BsmtFin Type 2": {"GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"},
    "Heating": {"Floor", "GasA", "GasW", "Grav", "OthW", "Wall"},
    "Heating QC": QUAL5,
    "Central Air": {"N", "Y"},
    "Electrical": {"SBrkr", "FuseA", "FuseF", "FuseP", "Mix"},
    "Kitchen Qual": QUAL5,
    "Functional": {"Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"},
    "Fireplace Qu": QUAL5 | {"NA"},
    "Garage Type": {"2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"},
    "Garage Finish": {"Fin", "RFn", "Unf", "NA"},
    "Garage Qual": QUAL5 | {"NA"},
    "Garage Cond": QUAL5 | {"NA"},
    "Paved Drive": {"Y", "P", "N"},
    "Pool QC": {"Ex", "Gd", "TA", "Fa", "NA"},
    "Fence": {"GdPrv", "MnPrv", "GdWo", "MnWw", "NA"},
    "Misc Feature": {"Elev", "Gar2", "Othr", "Shed", "TenC", "NA"},
    "Sale Type": {"WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"},
    "Sale Condition": {"Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"},
}
# The none-code of each structural column, which the cleaned file writes as ABSENT.
NONE_CODE = {c: "NA" for c, codes in ALLOWED.items() if "NA" in codes} | {"Mas Vnr Type": "None"}
ALLOWED = {c: codes | {ABSENT} if c in NONE_CODE else codes for c, codes in ALLOWED.items()}


def _failure(check, column, rule, mask, df, source) -> dict:
    n = int(mask.sum())
    ex = df.loc[mask, "Order"].head(5).tolist() if "Order" in df else []
    return dict(check=check, column=column, rule=rule, n=n, example_orders=ex, source=source,
                message=f"[{check}] {column}: {rule} -- {n} row(s), e.g. Order {ex} (source: {source})")


def _is_none(s: pd.Series, col: str) -> pd.Series:
    """True where a structural column says 'none', in either spelling (raw or cleaned)."""
    return s.isin([NONE_CODE[col], ABSENT])


# ---------------------------------------------------------------- checks
def check_schema(df: pd.DataFrame) -> list[dict]:
    """Return schema failures: missing or extra columns, wrong dtype families."""
    out = []
    cols = set(df.columns)
    for c in sorted(EXPECTED_COLUMNS - cols):
        out.append(dict(check="schema", column=c, rule="column missing", n=len(df), example_orders=[],
                        source=DICT, message=f"[schema] {c}: column missing"))
    for c in sorted(cols - EXPECTED_COLUMNS):
        out.append(dict(check="schema", column=c, rule="unexpected column", n=len(df), example_orders=[],
                        source=DICT, message=f"[schema] {c}: unexpected column"))
    for c in NUMERIC_COLUMNS:
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            out.append(dict(check="schema", column=c, rule=f"expected numeric, got {df[c].dtype}",
                            n=len(df), example_orders=[], source=DICT,
                            message=f"[schema] {c}: expected numeric, got {df[c].dtype}"))
    for c in TEXT_COLUMNS:
        if c in df and pd.api.types.is_numeric_dtype(df[c]):
            out.append(dict(check="schema", column=c, rule=f"expected text, got {df[c].dtype}",
                            n=len(df), example_orders=[], source=DICT,
                            message=f"[schema] {c}: expected text, got {df[c].dtype}"))
    return out


def check_ranges(df: pd.DataFrame) -> list[dict]:
    """One failure per column with values outside its range. NaN is not judged here."""
    out = []
    for c, (lo, hi, source) in RANGES.items():
        if c not in df or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        bad = df[c].notna() & ~df[c].between(lo, hi)
        if bad.any():
            vals = sorted(df.loc[bad, c].unique().tolist())[:5]
            out.append(_failure("range", c, f"outside [{lo}, {hi}], values {vals}", bad, df, source))
    return out


def check_allowed(df: pd.DataFrame) -> list[dict]:
    """One failure per categorical column with values not in the dictionary."""
    out = []
    for c, codes in ALLOWED.items():
        if c not in df:
            continue
        bad = df[c].notna() & ~df[c].isin(codes)
        if bad.any():
            found = df.loc[bad, c].value_counts().to_dict()
            out.append(_failure("allowed", c, f"codes not in dictionary {found}", bad, df,
                                DICT + f", {c}"))
    return out


def check_cross_column(df: pd.DataFrame) -> list[dict]:
    """Failures of rules that involve two or more columns. Rows with NaN in a rule's columns are skipped."""
    out = []
    d = df
    # Partial sales: dictionary says the home "was not completed when last assessed", so its
    # build/remodel year may be the year after the sale year. All other sales: not after the sale.
    grace = np.where(d["Sale Condition"].eq("Partial"), 1, 0)

    rules = [
        ("Year Remod/Add >= Year Built",
         d["Year Remod/Add"] < d["Year Built"],
         "dictionary: remodel date is the construction date if no remodeling"),
        ("Year Built <= Yr Sold (+1 for Partial sales)",
         d["Year Built"] > d["Yr Sold"] + grace,
         "a house cannot be built after it is sold; dictionary, Sale Condition 'Partial'"),
        ("Year Remod/Add <= Yr Sold (+1 for Partial sales)",
         d["Year Remod/Add"] > d["Yr Sold"] + grace,
         "a remodel recorded at sale cannot be later than the sale; dictionary, 'Partial'"),
        ("Garage Yr Blt <= Yr Sold (+1 for Partial sales)",
         d["Garage Yr Blt"] > d["Yr Sold"] + grace,
         "a garage cannot be built after the sale; dictionary, 'Partial'"),
        ("Mas Vnr Type is none <=> Mas Vnr Area == 0",
         d["Mas Vnr Type"].notna() & d["Mas Vnr Area"].notna()
         & (_is_none(d["Mas Vnr Type"], "Mas Vnr Type") != d["Mas Vnr Area"].eq(0)),
         "dictionary: Mas Vnr Type 'None' and Mas Vnr Area in square feet describe the same veneer"),
        ("Garage Type is none <=> Garage Area == 0",
         d["Garage Type"].notna() & d["Garage Area"].notna()
         & (_is_none(d["Garage Type"], "Garage Type") != d["Garage Area"].eq(0)),
         "dictionary: Garage Type 'NA No Garage'"),
        ("Garage Type not none => Garage Finish/Qual/Cond not none",
         d["Garage Type"].notna() & ~_is_none(d["Garage Type"], "Garage Type")
         & (_is_none(d["Garage Finish"], "Garage Finish") | _is_none(d["Garage Qual"], "Garage Qual")
            | _is_none(d["Garage Cond"], "Garage Cond")),
         "dictionary: 'NA No Garage' in Garage Finish/Qual/Cond only when there is no garage"),
        ("Bsmt Qual is none <=> Total Bsmt SF == 0",
         d["Bsmt Qual"].notna() & d["Total Bsmt SF"].notna()
         & (_is_none(d["Bsmt Qual"], "Bsmt Qual") != d["Total Bsmt SF"].eq(0)),
         "dictionary: Bsmt Qual 'NA No Basement'"),
        ("BsmtFin SF 1 + BsmtFin SF 2 + Bsmt Unf SF == Total Bsmt SF",
         d[["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF"]].notna().all(axis=1)
         & (d["BsmtFin SF 1"] + d["BsmtFin SF 2"] + d["Bsmt Unf SF"] != d["Total Bsmt SF"]),
         "dictionary definitions of the basement area columns"),
        ("1st Flr SF + 2nd Flr SF + Low Qual Fin SF == Gr Liv Area",
         d["1st Flr SF"] + d["2nd Flr SF"] + d["Low Qual Fin SF"] != d["Gr Liv Area"],
         "dictionary definitions of above-grade living area"),
        ("Pool QC is none <=> Pool Area == 0",
         d["Pool QC"].notna() & (_is_none(d["Pool QC"], "Pool QC") != d["Pool Area"].eq(0)),
         "dictionary: Pool QC 'NA No Pool'"),
        ("Fireplace Qu is none <=> Fireplaces == 0",
         d["Fireplace Qu"].notna() & (_is_none(d["Fireplace Qu"], "Fireplace Qu") != d["Fireplaces"].eq(0)),
         "dictionary: Fireplace Qu 'NA No Fireplace'"),
        ("Order and PID unique",
         d["Order"].duplicated(keep=False) | d["PID"].duplicated(keep=False),
         "dictionary: Order is the observation number, PID the parcel id"),
    ]
    for rule, mask, source in rules:
        mask = mask.fillna(False).astype(bool)
        if mask.any():
            out.append(_failure("cross", rule, "violated", mask, df, source))
    return out


def validate(df: pd.DataFrame) -> pd.DataFrame:
    errors = check_schema(df)
    if any(f["rule"] == "column missing" for f in errors):   # value rules need the columns
        raise ValidationError(errors)
    errors += check_ranges(df) + check_allowed(df) + check_cross_column(df)
    if errors:
        raise ValidationError(errors)
    return df


if __name__ == "__main__":
    path = sys.argv[1]
    raw = "--raw" in sys.argv
    expect_fail = "--expect-fail" in sys.argv
    df = read_raw(path) if raw else read_clean(path)
    try:
        validate(df)
        print(f"PASS {path}  ({len(df)} rows, {df.shape[1]} columns)")
        if expect_fail:
            sys.exit(f"ERROR: {path} was expected to FAIL validation but passed.")
    except ValidationError as e:
        print(f"FAIL {path}: {len(e.failures)} rule(s) failed\n{e}")
        if raw:
            out = Path("reports/results")
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(e.failures).drop(columns="message").to_csv(out / "validation_raw.csv", index=False)
        if expect_fail:
            print("(expected: the raw file must fail validation)")
            sys.exit(0)
        sys.exit(1)
