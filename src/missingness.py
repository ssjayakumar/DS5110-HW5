"""Part 5: evidence about the missingness mechanism (notebook §4.1 pattern).

For each column with true missing values, split the houses into "value missing" and
"value present" and compare the two groups on columns we always observe. Under MCAR the
groups are two random samples of the same houses, so they should not differ beyond chance.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from io_utils import md_table, read_clean, write_result

TARGETS = ["Lot Frontage", "Mas Vnr Area"]
NUMERIC_OBS = ["Lot Area", "SalePrice", "Year Built", "Gr Liv Area"]
CATEGORICAL_OBS = ["Lot Config", "Neighborhood", "Bldg Type"]


def compare_groups(df: pd.DataFrame, col: str, observed_cols: list[str]) -> pd.DataFrame:
    """Compare rows with and without `col` on numeric observed columns. Return a summary table.

    Medians (robust to the skew in areas and prices), the standardized mean difference, and a
    Mann-Whitney U test of "same distribution in both groups".
    """
    miss = df[col].isna()
    rows = []
    for c in observed_cols:
        a, b = df.loc[miss, c].dropna(), df.loc[~miss, c].dropna()
        pooled_sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({
            "observed column": c,
            "median, missing": a.median(), "median, present": b.median(),
            "mean, missing": a.mean(), "mean, present": b.mean(),
            "std. mean diff": (a.mean() - b.mean()) / pooled_sd,
            "Mann-Whitney p": stats.mannwhitneyu(a, b).pvalue,
        })
    return pd.DataFrame(rows)


def compare_categories(df: pd.DataFrame, col: str, cat: str, top: int = 6) -> tuple[pd.DataFrame, float]:
    """Share missing within each level of a categorical column, plus a chi-square test p-value."""
    miss = df[col].isna()
    tab = pd.crosstab(df[cat], miss)
    p = stats.chi2_contingency(tab).pvalue
    by = (df.assign(_m=miss).groupby(cat)["_m"].agg(n="size", missing="sum", share_missing="mean")
          .sort_values("share_missing", ascending=False))
    return by.head(top).reset_index(), p


def run(df: pd.DataFrame) -> None:
    summary = []
    for col in TARGETS:
        miss = df[col].isna()
        summary.append(dict(column=col, missing=int(miss.sum()), share=miss.mean()))
        # Notebook §4.1: group by "is missing" and compare means of observed columns.
        nb = (df.groupby(df[col].isna())[NUMERIC_OBS].mean()
              .rename_axis(f"{col} missing").reset_index())
        num = compare_groups(df, col, NUMERIC_OBS)
        slug = col.lower().replace(" ", "_")
        parts = [md_table(nb, floatfmt="{:.0f}"), md_table(num, floatfmt="{:.3g}")]
        cat_rows = []
        for cat in CATEGORICAL_OBS:
            by, p = compare_categories(df, col, cat)
            cat_rows.append(dict(observed=cat, chi2_p=p, overall_share=miss.mean(),
                                 highest=", ".join(f"{r[cat]} {r['share_missing']:.0%} (n={r['n']})"
                                                   for _, r in by.head(3).iterrows())))
        parts.append(md_table(pd.DataFrame(cat_rows), floatfmt="{:.3g}"))
        write_result(f"mech_{slug}", "\n\n".join(parts))
        print(f"{col}: {miss.sum()} missing ({miss.mean():.1%})")
        print(num.round(3).to_string(index=False))
        print(pd.DataFrame(cat_rows).to_string(index=False))

    write_result("mech_share", md_table(pd.DataFrame(summary), floatfmt="{:.4f}"))

    # Extra evidence for Lot Frontage: missing share by Lot Config (the strongest signal).
    by, _ = compare_categories(df, "Lot Frontage", "Lot Config", top=10)
    write_result("mech_lotconfig", md_table(by, floatfmt="{:.3f}"))

    # Mas Vnr Area: are the missing rows concentrated in recent construction?
    mv = df["Mas Vnr Area"].isna()
    yb = df.loc[mv, "Year Built"].describe()[["min", "50%", "max"]]
    write_result("mech_masvnr_years",
                 f"Year Built of the {int(mv.sum())} houses with missing veneer: min {yb['min']:.0f}, "
                 f"median {yb['50%']:.0f}, max {yb['max']:.0f}; "
                 f"{(df.loc[mv, 'Year Built'] >= 2000).mean():.0%} built in 2000 or later "
                 f"(all houses: {(df['Year Built'] >= 2000).mean():.0%}).")


if __name__ == "__main__":
    run(read_clean(sys.argv[1]))
