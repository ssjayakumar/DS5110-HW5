# DS5110 HW 5. Run everything with:  make all
# Recipes use only plain commands (no shell "!", "||", mkdir -p or rm -rf), so the same
# Makefile runs with GNU make on Linux/macOS and with GnuWin32 make on Windows.
PY    := python
RAW   := data/raw/AmesHousing.txt
CLEAN := data/processed/ames_clean.csv

.PHONY: all validate clean

all: validate reports/report.md

$(RAW):
	$(PY) src/download.py $(RAW)

# clean.py writes the cleaned file, then the Part 3-4 tables (so the tables are newer).
$(CLEAN): $(RAW) src/clean.py src/validate.py src/io_utils.py
	$(PY) src/clean.py $(RAW) $(CLEAN)

# Rebuild the Part 3-4 tables if they were deleted on their own.
reports/results/changes.md: $(CLEAN)
	$(PY) src/clean.py $(RAW) $(CLEAN)

# Succeeds only if validation FAILS on raw data and PASSES on cleaned data.
# --expect-fail inverts the exit code (the starter's "! python ..."), portable to Windows.
# The raw run also saves its failure table to reports/results/validation_raw.csv.
validate: $(RAW) $(CLEAN)
	$(PY) src/validate.py $(RAW) --raw --expect-fail
	$(PY) src/validate.py $(CLEAN)

# Depends on validate (phony) so the report is always rebuilt from fresh results.
reports/report.md: validate $(CLEAN) reports/results/changes.md src/missingness.py src/impute.py src/evaluate.py src/report.py reports/report_template.md
	$(PY) src/missingness.py $(CLEAN)
	$(PY) src/impute.py $(CLEAN)
	$(PY) src/evaluate.py $(CLEAN)
	$(PY) src/report.py

clean:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('data/processed', 'reports/figures', 'reports/results')]"
