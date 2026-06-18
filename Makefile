RESULTS_PATH ?= results_v2

setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt -r requirements_dev.txt

lint:
	flake8

# ── Dados ──────────────────────────────────────────────────────────────────────

download-cumida:
	python scripts/download_cumida_datasets.py

build-synthetic:
	python scripts/synthetic_datasets.py

build-xor:
	python scripts/xor_dataset.py

# ── Pipeline principal (compare_deep) ─────────────────────────────────────────

run-compare: run-select run-scoring run-stability run-times

run-select:
	python src/main.py select \
		--presets compare_deep \
		--datasets_path datasets \
		--results-path $(RESULTS_PATH) \
		--selection-filename compare-selection \
		--workers 1 \
		-vv

run-scoring:
	python src/main.py scoring \
		--datasets_path datasets \
		--results-path $(RESULTS_PATH) \
		--selection-filename compare-selection.csv \
		--scoring-filename compare-scoring \
		-vv

run-stability:
	python src/main.py stability \
		--results-path $(RESULTS_PATH) \
		--selection-filename compare-selection.csv \
		--stability-filename compare-stability \
		--workers 1 \
		-vv

run-times:
	python src/main.py times \
		--results-path $(RESULTS_PATH) \
		--selection-filename compare-selection.csv \
		-vv

# ── Análise de resultados ──────────────────────────────────────────────────────

analyze:
	python aux/analyze_results.py --results-path $(RESULTS_PATH)

# ── Diagnósticos ───────────────────────────────────────────────────────────────

entropy-check:
	python aux/channel_attention_entropy.py

# ── Smoke tests ────────────────────────────────────────────────────────────────

run-test:
	python src/main.py all \
		--presets test_algorithms \
		--datasets_path datasets \
		--results-path results_test \
		--workers 1 \
		-vv

# ── Legado ─────────────────────────────────────────────────────────────────────

run:
	python src/main.py all -p default -n 1 -vv

run-reduced:
	python src/main.py all -p reduced -n 31 -vv
