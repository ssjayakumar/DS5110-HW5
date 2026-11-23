# HW 5: Ames Housing quality audit

DS5110 Essentials of Data Science, Fall 2026. Audit of the Ames Housing data (De Cock, 2011):
provenance, validation as code, sentinels, keys, missingness mechanism, imputation and
leakage-safe evaluation. The report is `reports/report.md`.

## How to run

```
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt
make all
```

On Windows, run these in Git Bash with GNU make installed (e.g. `winget install GnuWin32.Make`,
then add `C:\Program Files (x86)\GnuWin32\bin` to PATH). The Makefile uses no shell-only syntax,
so it runs the same with GNU make on Linux, macOS and Windows.

Requires Python 3.10 or newer (tested on 3.10, 3.12 and 3.14). `make all` downloads the raw file if it is missing,
cleans it, checks that validation **fails on the raw file and passes on the cleaned file**, runs
Parts 5 to 7, and regenerates `reports/report.md` from `reports/report_template.md`.
`make validate` runs only the validation gate (`validate.py --expect-fail` on the raw file, then a normal
run on the cleaned file); `make clean` removes all generated outputs
(the raw file is never deleted or edited).

| Step | Script | Output |
| --- | --- | --- |
| Part 1 | `src/download.py` | `data/raw/AmesHousing.txt` (never overwritten) |
| Part 2 | `src/validate.py` | pass/fail, `reports/results/validation_raw.csv` |
| Parts 3-4 | `src/clean.py` | `data/processed/ames_clean.csv`, `data/processed/neighborhoods.csv` |
| Part 5 | `src/missingness.py` | `reports/results/mech_*.md` |
| Part 6 | `src/impute.py` | `reports/figures/lot_frontage_*.png`, cost tables |
| Part 7 | `src/evaluate.py` | RMSE tables for variants A, B, C |
| Part 8 | `src/report.py` | `reports/report.md` |

`src/io_utils.py` holds the one way every script reads the files: `keep_default_na=False`,
`na_values=[""]` and PID as text, so "NA", "None" and the leading zeros of PID survive.

## Provenance (Part 1)

| Item | Value |
| --- | --- |
| Source URL | https://jse.amstat.org/v19n3/decock/AmesHousing.txt (the data file linked from the JSE article; `DATA_URL` in `src/download.py`) |
| Downloaded on | 2026-11-23, from the JSE URL above by `src/download.py` (`make all`) |
| Checksum | MD5 `c2997eb69ab4bf946b9fbdb7e6805e36`. `download.py` falls back to a public mirror (`https://raw.githubusercontent.com/rasbt/machine-learning-book/main/ch09/AmesHousing.txt`) only if JSE is unreachable; that mirror's file has the same MD5. |
| Citation | De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, 19(3). |
| Data dictionary | https://jse.amstat.org/v19n3/decock/DataDocumentation.txt |
| Terms of use | JSE leaves copyright with the contributing author, so the data may be used with citation for teaching and study, but publishing it requires the author's consent (https://jse.amstat.org/jse_users.htm). |
| Rows, columns on arrival | 2,930 rows, 82 columns (checked by `download.py` before saving) |

`download.py` refuses to overwrite an existing raw copy: it returns early if the file exists, and it
opens the file in exclusive-create mode (`"xb"`), so it can never replace one.

## Features used in Part 7

`NUMERIC_FEATURES` (12 columns, in `src/evaluate.py`):

| Column | Why |
| --- | --- |
| Lot Frontage | The column under study. It must be in X for the imputation variants to differ. |
| Lot Area | Land size; also the strongest predictor of Lot Frontage. |
| Overall Qual | Single strongest predictor of price (material and finish, 1-10). |
| Overall Cond | Condition, separate from quality. |
| Year Built | Age of the house. |
| Gr Liv Area | Above-grade living area, the main size measure. |
| Total Bsmt SF | Basement size (true 0 when there is no basement). |
| Full Bath, Half Bath | Bathrooms above grade. |
| TotRms AbvGrd | Room count, layout beyond area. |
| Fireplaces | Amenity with a clean true-zero coding. |
| Garage Cars | Garage capacity (true 0 when there is no garage). |

All are numeric with no structural NaN. Year Remod/Add, Garage Yr Blt and Mas Vnr Area are left out
on purpose: they contain values this audit set to NaN (contradictions) or structural NaN (no garage),
and the point of Part 7 is to vary only the Lot Frontage fill. Two rows with a true unknown in another
feature (Order 1342, 2237) are dropped in Part 7, so 2,928 rows are used. In variant C, `Lot Config`
and `Neighborhood` are passed through X only for the imputer, which drops them before the model.

## AI assistance

| Tool | Used for | Files |
| --- | --- | --- |
| Claude (Anthropic, Claude Opus 5.5), via claude.ai | Drafting the pipeline code, the Makefile changes, the report template and this README from the assignment and the starter repo; exploring the data and dictionary; finding a mirror of the raw file. | `src/*.py`, `Makefile`, `reports/report_template.md`, `README.md` |

I reviewed, ran and checked every file, and I can explain every line. All numbers in the
report are produced by the code, not written by hand.
