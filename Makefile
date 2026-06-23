RESULTS_PATH ?= results_v2
RESULTS_PATH_V3 ?= results_v3
PYTHON ?= python3

setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt -r requirements_dev.txt

lint:
	flake8

# ── Dados ──────────────────────────────────────────────────────────────────────

download-cumida:
	$(PYTHON) scripts/download_cumida_datasets.py

build-synthetic:
	$(PYTHON) scripts/synthetic_datasets.py

build-xor:
	$(PYTHON) scripts/xor_dataset.py

# ── Pipeline principal (compare_deep) ─────────────────────────────────────────

run-compare: run-select run-scoring run-stability run-times

run-select:
	$(PYTHON) src/main.py select \
		--presets compare_deep \
		--datasets_path datasets \
		--results-path $(RESULTS_PATH) \
		--selection-filename compare-selection \
		--workers 1 \
		-vv

run-scoring:
	$(PYTHON) src/main.py scoring \
		--datasets_path datasets \
		--results-path $(RESULTS_PATH) \
		--selection-filename compare-selection.csv \
		--scoring-filename compare-scoring \
		-vv

run-stability:
	$(PYTHON) src/main.py stability \
		--results-path $(RESULTS_PATH) \
		--selection-filename compare-selection.csv \
		--stability-filename compare-stability \
		--workers 1 \
		-vv

run-times:
	$(PYTHON) src/main.py times \
		--results-path $(RESULTS_PATH) \
		--selection-filename compare-selection.csv \
		-vv

# ── Pipeline v3 (compare_deep_multi — Prostate + Breast) ─────────────────────

run-v3: run-v3-select run-v3-scoring run-v3-stability run-v3-times

run-v3-select:
	$(PYTHON) src/main.py select \
		--presets compare_deep_multi \
		--datasets_path datasets \
		--results-path $(RESULTS_PATH_V3) \
		--selection-filename compare-selection \
		--workers 1 \
		-vv

run-v3-scoring:
	$(PYTHON) src/main.py scoring \
		--datasets_path datasets \
		--results-path $(RESULTS_PATH_V3) \
		--selection-filename compare-selection.csv \
		--scoring-filename compare-scoring \
		-vv

run-v3-stability:
	$(PYTHON) src/main.py stability \
		--results-path $(RESULTS_PATH_V3) \
		--selection-filename compare-selection.csv \
		--stability-filename compare-stability \
		--dataset Prostate_GSE6919_U95C \
		--workers 1 \
		-vv

run-v3-times:
	$(PYTHON) src/main.py times \
		--results-path $(RESULTS_PATH_V3) \
		--selection-filename compare-selection.csv \
		--dataset Prostate_GSE6919_U95C \
		-vv

run-v3-stats:
	$(PYTHON) src/main.py stats \
		--results-path $(RESULTS_PATH_V3) \
		--scoring-filename compare-scoring \
		--stats-filename compare-stats \
		--dataset Prostate_GSE6919_U95C \
		-vv

analyze-v3:
	$(PYTHON) aux/analyze_results.py --results-path $(RESULTS_PATH_V3) --dataset Prostate_GSE6919_U95C

# ── Análise de resultados ──────────────────────────────────────────────────────

analyze:
	$(PYTHON) aux/analyze_results.py --results-path $(RESULTS_PATH)

analyze-cumida:
	$(PYTHON) scripts/analyze_cumida.py \
		--datasets-path datasets/cumida \
		--results-path $(RESULTS_PATH)

# ── Diagnósticos ───────────────────────────────────────────────────────────────

entropy-check:
	$(PYTHON) aux/channel_attention_entropy.py

# ── Smoke tests ────────────────────────────────────────────────────────────────

run-test:
	$(PYTHON) src/main.py all \
		--presets test_algorithms \
		--datasets_path datasets \
		--results-path results_test \
		--workers 1 \
		-vv

# ── Legado ─────────────────────────────────────────────────────────────────────

run:
	$(PYTHON) src/main.py all -p default -n 1 -vv

run-reduced:
	$(PYTHON) src/main.py all -p reduced -n 31 -vv
