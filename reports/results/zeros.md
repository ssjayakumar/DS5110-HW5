| column | zeros | none_column | argument |
| --- | --- | --- | --- |
| Mas Vnr Area | 1748 | Mas Vnr Type | True zero: no veneer has 0 sq ft. The 'none' itself is in Mas Vnr Type. |
| Total Bsmt SF | 79 | Bsmt Qual | True zero: 0 sq ft exactly when Bsmt Qual says No Basement (rule checked). |
| BsmtFin SF 1 | 930 | BsmtFin Type 1 | True zero: unfinished or absent basement has 0 finished sq ft. |
| BsmtFin SF 2 | 2578 | BsmtFin Type 2 | True zero, same as BsmtFin SF 1. |
| Bsmt Unf SF | 244 | Bsmt Qual | True zero: fully finished or absent basement. |
| Bsmt Full Bath | 1707 | Bsmt Qual | True zero count; for no-basement houses it is also structural. |
| Bsmt Half Bath | 2753 | Bsmt Qual | True zero count, as above. |
| 2nd Flr SF | 1678 | House Style | True zero: 1-story houses have no second floor. |
| Low Qual Fin SF | 2890 | - | True zero: no low-quality finished area. |
| Full Bath | 12 | - | True zero count (above grade only; such houses have basement baths). |
| Half Bath | 1843 | - | True zero count. |
| Bedroom AbvGr | 8 | - | True zero: the dictionary excludes basement bedrooms from this count. |
| Kitchen AbvGr | 3 | - | Plausible true zero (above grade only), but Kitchen Qual is still rated: flagged, kept. |
| Fireplaces | 1422 | Fireplace Qu | True zero: exactly when Fireplace Qu says No Fireplace (rule checked). |
| Garage Cars | 157 | Garage Type | True zero: no garage holds 0 cars (Garage Type = No Garage). |
| Garage Area | 157 | Garage Type | True zero: 0 sq ft exactly when there is no garage (rule checked). |
| Wood Deck SF | 1526 | - | True zero: no deck. No separate 'none' column exists. |
| Open Porch SF | 1300 | - | True zero: no porch of this type. |
| Enclosed Porch | 2471 | - | True zero: no porch of this type. |
| 3Ssn Porch | 2893 | - | True zero: no porch of this type. |
| Screen Porch | 2674 | - | True zero: no porch of this type. |
| Pool Area | 2917 | Pool QC | True zero: exactly when Pool QC says No Pool (rule checked). |
| Misc Val | 2827 | Misc Feature | Mostly structural (no feature). 3 rows have a feature with $0 value: kept as recorded. |
