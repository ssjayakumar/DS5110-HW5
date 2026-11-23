# DS5110 HW 5. Run everything with:  make all
PY    := python
RAW   := data/raw/AmesHousing.txt
CLEAN := data/processed/ames_clean.csv

.PHONY: all validate clean

all: validate reports/report.md

$(RAW):
	$(PY) src/download.py $(RAW)

$(CLEAN): $(RAW) src/clean.py src/validate.py src/io_utils.py
	mkdir -p data/processed reports/results
	$(PY) src/clean.py $(RAW) $(CLEAN)

# clean.py also writes the Part 3-4 tables; rebuild them if they were deleted on their own.
reports/results/changes.md: $(CLEAN)
	@test -f $@ || $(PY) src/clean.py $(RAW) $(CLEAN)

# Succeeds only if validation FAILS on raw data and PASSES on cleaned data.
# (The raw run also saves its failure table to reports/results/validation_raw.csv.)
validate: $(RAW) $(CLEAN)
	! $(PY) src/validate.py $(RAW) --raw
	$(PY) src/validate.py $(CLEAN)

# Depends on validate (phony) so the report is always rebuilt from fresh results.
reports/report.md: validate $(CLEAN) reports/results/changes.md src/missingness.py src/impute.py src/evaluate.py src/report.py reports/report_template.md
	mkdir -p reports/figures reports/results
	$(PY) src/missingness.py $(CLEAN)
	$(PY) src/impute.py $(CLEAN)
	$(PY) src/evaluate.py $(CLEAN)
	$(PY) src/report.py

clean:
	rm -rf data/processed reports/figures reports/results
