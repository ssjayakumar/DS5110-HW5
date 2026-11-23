"""Part 8: build reports/report.md from reports/report_template.md and the pipeline's results.

The narrative is written by hand in the template. Every table and every number that the
pipeline computes is inserted from reports/results/, so the report always matches the code.
Placeholders look like {{name}} and are replaced by reports/results/name.md.
"""
import re
import sys
from pathlib import Path

import pandas as pd

from io_utils import RESULTS, md_table

TEMPLATE = Path("reports/report_template.md")
OUT = Path("reports/report.md")

# What was done about each failure on the raw file (Part 2 table). Keyed by (check, column).
ACTIONS = {
    ("range", "Garage Yr Blt"): "Set to NaN (2207 is impossible; 2007 is a guess, not a fix).",
    ("allowed", "MS Zoning"): "Mapped to dictionary codes C, I, A (explicit map in clean.py, logged).",
    ("allowed", "Neighborhood"): "Mapped NAmes -> Names (dictionary code for North Ames), logged.",
    ("allowed", "Bldg Type"): "Mapped Duplex->Duplx, Twnhs->TwnhsI, 2fmCon->2FmCon, logged.",
    ("allowed", "Exterior 2nd"): "Mapped CmentBd->CemntBd, Wd Shng->WdShing, Brk Cmn->BrkComm, logged.",
    ("allowed", "Sale Type"): "Stripped the trailing space ('WD ' -> 'WD').",
    ("cross", "Year Remod/Add >= Year Built"): "Year Remod/Add set to NaN (cannot tell which year is wrong).",
    ("cross", "Year Remod/Add <= Yr Sold (+1 for Partial sales)"): "Year Remod/Add set to NaN.",
    ("cross", "Garage Yr Blt <= Yr Sold (+1 for Partial sales)"): "Same row as the 2207 range failure: NaN.",
    ("cross", "Mas Vnr Type is none <=> Mas Vnr Area == 0"): "Both columns set to NaN in these rows.",
    ("cross", "Garage Type not none => Garage Finish/Qual/Cond not none"):
        "Garage exists, so 'No Garage' is wrong: Qual/Cond set to NaN (unknown).",
}


def validation_table() -> str:
    f = pd.read_csv(RESULTS / "validation_raw.csv")
    f["what we did"] = [ACTIONS.get((c, col), "(no action recorded)") for c, col in zip(f["check"], f["column"])]
    f["rule / column"] = f["column"]
    f["finding"] = f["rule"]
    return md_table(f[["check", "rule / column", "finding", "n", "what we did"]])


def main() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    fragments = {"validation_table": validation_table()}

    def sub(m):
        name = m.group(1)
        if name in fragments:
            return fragments[name]
        path = RESULTS / f"{name}.md"
        if not path.exists():
            sys.exit(f"report.py: missing result {path} for placeholder {{{{{name}}}}}")
        return path.read_text(encoding="utf-8").strip()

    OUT.write_text(re.sub(r"\{\{(\w+)\}\}", sub, text), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
