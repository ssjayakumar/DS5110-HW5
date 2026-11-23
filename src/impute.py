"""Part 6: at least two strategies for Lot Frontage, and their costs (notebook §5 pattern).

Strategies
  1. median       : one number for every missing lot (simple fill)
  2. nbhd_median  : median Lot Frontage of the house's Neighborhood (uses another column)
  3. conditional  : regression of log(Lot Frontage) on log(Lot Area), Lot Config and
                    Neighborhood, the columns Part 5 found missingness depends on. CHOSEN.

Strategies 2 and 3 are written as scikit-learn transformers (fit learns from the training
rows only, transform fills), so Part 7 can put them inside a Pipeline without leakage.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from io_utils import FIGURES, md_table, read_clean, write_result

TARGET = "Lot Frontage"
OBSERVED = "#2a78d6"   # categorical slot 1 (dataviz default palette)
IMPUTED = "#eb6834"    # categorical slot 2


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill `column` with the median of its group; groups unseen in fit get the overall median.

    Medians are computed in fit() from the rows it is given, never in transform(), so inside a
    Pipeline each CV fold learns its medians from its own training rows only.
    """

    def __init__(self, column=TARGET, group="Neighborhood", drop=()):
        self.column, self.group, self.drop = column, group, drop

    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.group)[self.column].median()
        self.overall_ = X[self.column].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X[self.group].map(self.medians_).fillna(self.overall_)
        X[self.column] = X[self.column].fillna(fill)
        return X.drop(columns=list(self.drop))


class ConditionalFrontageImputer(BaseEstimator, TransformerMixin):
    """Fill Lot Frontage from log(Lot Area), Lot Config and Neighborhood (the chosen strategy).

    fit(): on rows where Lot Frontage is observed, fit
        log(Lot Frontage) ~ log(Lot Area) + one-hot(Lot Config) + one-hot(Neighborhood)
    transform(): for missing rows, fill exp(prediction). Unseen categories get all-zero dummies.
    `drop` removes the helper text columns afterwards so the model sees only numeric features.
    """

    def __init__(self, column=TARGET, area="Lot Area", cats=("Lot Config", "Neighborhood"), drop=()):
        self.column, self.area, self.cats, self.drop = column, area, cats, drop

    def _design(self, X):
        dummies = self.encoder_.transform(X[list(self.cats)])
        return np.column_stack([np.log(X[self.area].to_numpy()), dummies])

    def fit(self, X, y=None):
        obs = X[X[self.column].notna()]
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(obs[list(self.cats)])
        self.model_ = LinearRegression().fit(self._design(obs), np.log(obs[self.column]))
        return self

    def transform(self, X):
        X = X.copy()
        miss = X[self.column].isna()
        if miss.any():
            X.loc[miss, self.column] = np.exp(self.model_.predict(self._design(X[miss])))
        return X.drop(columns=list(self.drop))


def fill_median(df: pd.DataFrame) -> pd.Series:
    return df[TARGET].fillna(df[TARGET].median())


STRATEGIES = {
    "median": lambda d: fill_median(d),
    "nbhd_median": lambda d: GroupMedianImputer().fit(d).transform(d)[TARGET],
    "conditional": lambda d: ConditionalFrontageImputer().fit(d).transform(d)[TARGET],
}
LABELS = {"median": "Median fill", "nbhd_median": "Neighborhood median",
          "conditional": "Conditional (log Lot Area + Lot Config + Neighborhood)"}


def cost_table(before: pd.Series, after: pd.Series, other: pd.Series) -> pd.Series:
    """std before/after, correlation with `other` before/after (before = observed rows only)."""
    return pd.Series({
        "mean before": before.mean(), "mean after": after.mean(),
        "std before": before.std(), "std after": after.std(),
        "std change %": 100 * (after.std() / before.std() - 1),
        "corr Lot Area before": before.corr(other), "corr Lot Area after": after.corr(other),
        "corr log-log before": np.log(before).corr(np.log(other)),
        "corr log-log after": np.log(after).corr(np.log(other)),
    })


def masked_error(df: pd.DataFrame, seed: int = 5110, frac: float = 0.2) -> pd.DataFrame:
    """Hide 20% of the observed values, fill them, and measure the error. Supporting evidence only."""
    obs = df[df[TARGET].notna()]
    hide = obs.sample(frac=frac, random_state=seed).index
    d = df.copy()
    d.loc[hide, TARGET] = np.nan
    rows = []
    for name, f in STRATEGIES.items():
        err = f(d).loc[hide] - df.loc[hide, TARGET]
        rows.append(dict(strategy=name, hidden=len(hide), RMSE_ft=np.sqrt((err ** 2).mean()),
                         median_abs_err_ft=err.abs().median()))
    return pd.DataFrame(rows)


def histogram(df: pd.DataFrame, filled: pd.Series, name: str) -> None:
    miss = df[TARGET].isna()
    bins = np.arange(0, 320, 5)
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
    ax.hist([filled[~miss], filled[miss]], bins=bins, stacked=True, color=[OBSERVED, IMPUTED],
            edgecolor="white", linewidth=0.6,
            label=[f"Observed (n={(~miss).sum()})", f"Imputed (n={miss.sum()})"])
    ax.set(title=f"Lot Frontage after fill: {LABELS[name]}", xlabel="Lot Frontage (ft)",
           ylabel="Houses", xlim=(0, 320))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"lot_frontage_{name}.png")
    plt.close(fig)


if __name__ == "__main__":
    df = read_clean(sys.argv[1])
    FIGURES.mkdir(parents=True, exist_ok=True)
    before = df[TARGET]
    costs = {}
    for name, f in STRATEGIES.items():
        filled = f(df)
        assert filled.notna().all(), name
        costs[name] = cost_table(before, filled, df["Lot Area"])
        histogram(df, filled, name)
    costs = pd.DataFrame(costs).T.rename_axis("strategy")
    print(costs.round(3).to_string())
    write_result("impute_costs", md_table(costs, index=True))
    err = masked_error(df)
    print(err.round(2).to_string(index=False))
    write_result("impute_masked", md_table(err, floatfmt="{:.2f}"))
