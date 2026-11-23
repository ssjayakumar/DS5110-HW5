"""Part 7: variants A, B, C with the given model (notebook §6.1 to §6.3).

A  median Lot Frontage within each SalePrice quintile, computed on the full table  (target leakage)
B  Part 6 strategy (ConditionalFrontageImputer) fitted on the full table, then CV  (test-fold leakage)
C  the same imputer inside a Pipeline, refitted on each training fold             (leakage-safe)
Reference: Lot Frontage dropped entirely, to show how much the column matters at all.

The model sees exactly NUMERIC_FEATURES. Lot Config and Neighborhood are carried in X only so the
imputer can use them; the imputer drops them before LinearRegression.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline

from impute import ConditionalFrontageImputer
from io_utils import md_table, read_clean, write_result

NUMERIC_FEATURES: list[str] = [   # 12 columns, including "Lot Frontage" (reasons in README)
    "Lot Frontage", "Lot Area", "Overall Qual", "Overall Cond", "Year Built", "Gr Liv Area",
    "Total Bsmt SF", "Full Bath", "Half Bath", "TotRms AbvGrd", "Fireplaces", "Garage Cars",
]
HELPERS = ["Lot Config", "Neighborhood"]
CV = KFold(n_splits=5, shuffle=True, random_state=5110)


def rmse(estimator, X: pd.DataFrame, y: pd.Series) -> float:
    return -cross_val_score(estimator, X, y, cv=CV, scoring="neg_root_mean_squared_error").mean()


def fold_scores(estimator, X, y) -> np.ndarray:
    return -cross_val_score(estimator, X, y, cv=CV, scoring="neg_root_mean_squared_error")


def variant_a(df: pd.DataFrame) -> pd.DataFrame:
    """§6.1: fill with the median Lot Frontage of the house's SalePrice quintile (uses y!)."""
    d = df.copy()
    q = pd.qcut(d["SalePrice"], 5, labels=False)
    d["Lot Frontage"] = d["Lot Frontage"].fillna(d.groupby(q)["Lot Frontage"].transform("median"))
    return d


def variant_b(df: pd.DataFrame) -> pd.DataFrame:
    """§6.2: the Part 6 imputer fitted once on all 2,928 rows, test folds included."""
    return ConditionalFrontageImputer().fit(df).transform(df)


if __name__ == "__main__":
    df = read_clean(sys.argv[1])
    # Other features have 1-2 true unknowns (Order 1342: basement; 2237: garage). Drop those rows
    # so Lot Frontage is the only column the variants handle differently.
    other = [c for c in NUMERIC_FEATURES if c != "Lot Frontage"]
    dropped = df.loc[df[other].isna().any(axis=1), "Order"].tolist()
    df = df.dropna(subset=other).reset_index(drop=True)
    y = np.log(df["SalePrice"])
    model = LinearRegression()

    A = variant_a(df)[NUMERIC_FEATURES]
    B = variant_b(df)[NUMERIC_FEATURES]
    C_pipe = make_pipeline(ConditionalFrontageImputer(drop=HELPERS), LinearRegression())
    C_X = df[NUMERIC_FEATURES + HELPERS]
    R = df[[c for c in NUMERIC_FEATURES if c != "Lot Frontage"]]

    rows = []
    for name, est, X, fill in [
        ("A", model, A, "SalePrice-quintile median, full table (6.1)"),
        ("B", model, B, "conditional imputer fitted on full table before CV (6.2)"),
        ("C", C_pipe, C_X, "conditional imputer inside Pipeline (6.3)"),
        ("ref", model, R, "Lot Frontage dropped (not a variant; for scale)"),
    ]:
        s = fold_scores(est, X, y)
        rows.append(dict(variant=name, fill=fill, RMSE_log=s.mean(), fold_sd=s.std(ddof=1),
                         **{f"fold{i+1}": v for i, v in enumerate(s)}))
    res = pd.DataFrame(rows)
    print(res.round(5).to_string(index=False))

    a, b, c = res.set_index("variant").loc[["A", "B", "C"], "RMSE_log"]
    # How far apart are B's and C's filled values on the test folds? (explains the B-C size)
    diffs = []
    for tr, te in CV.split(df):
        imp = ConditionalFrontageImputer().fit(df.iloc[tr])
        miss = df.iloc[te]["Lot Frontage"].isna().to_numpy()
        fold_fill = imp.transform(df.iloc[te])["Lot Frontage"].to_numpy()[miss]
        full_fill = B.iloc[te]["Lot Frontage"].to_numpy()[miss]
        diffs.append(np.abs(fold_fill - full_fill))
    diffs = np.concatenate(diffs)

    # Diagnostic (not one of the graded variants): the same three fills with Lot Frontage as the
    # ONLY feature, where nothing else can mask what the fill carries; and how strongly each
    # variant's filled values track the target on the rows that were missing.
    miss = df["Lot Frontage"].isna()
    diag = []
    for name, est, X in [
        ("A", model, A[["Lot Frontage"]]),
        ("B", model, B[["Lot Frontage"]]),
        ("C", make_pipeline(ConditionalFrontageImputer(drop=HELPERS), LinearRegression()),
         df[["Lot Frontage", "Lot Area"] + HELPERS]),
    ]:
        if name == "C":   # keep only Lot Frontage after the fill: drop Lot Area too
            est = make_pipeline(ConditionalFrontageImputer(drop=HELPERS + ["Lot Area"]), LinearRegression())
        fill = (A if name == "A" else B)["Lot Frontage"][miss]
        diag.append(dict(variant=name, RMSE_log_LotFrontage_only=rmse(est, X, y),
                         corr_fill_vs_logprice_on_missing_rows=np.corrcoef(fill, y[miss])[0, 1]))
    diag = pd.DataFrame(diag)
    print(diag.round(4).to_string(index=False))
    write_result("eval_diag", md_table(diag, floatfmt="{:.4f}"))

    write_result("eval_scores", md_table(res[["variant", "fill", "RMSE_log", "fold_sd"]], floatfmt="{:.5f}"))
    write_result("eval_folds", md_table(res.drop(columns=["fill", "fold_sd"]), floatfmt="{:.5f}"))
    write_result("eval_facts", "\n".join([
        f"- Rows used: {len(df)} (dropped Order {dropped}: unknown values in other features).",
        f"- A - C = {a - c:+.5f}; B - C = {b - c:+.6f} ({100 * (b - c) / c:+.3f}% of C).",
        f"- Lot Frontage missing in {int(df['Lot Frontage'].isna().sum())} rows "
        f"({df['Lot Frontage'].isna().mean():.1%}).",
        f"- B vs C fills on test-fold rows: mean |difference| {diffs.mean():.2f} ft, "
        f"max {diffs.max():.2f} ft (fills themselves average "
        f"{B['Lot Frontage'][df['Lot Frontage'].isna()].mean():.1f} ft).",
    ]))
