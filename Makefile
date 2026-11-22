PY ?= .venv/bin/python
NICE = nice -n 10   # benchmarks run at low priority so the machine stays responsive

.PHONY: setup demo serve test lint bench bench-quick clean

setup:            ## create venv and install (no system-wide changes)
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"
	./scripts/macos_link_libomp.sh $(PY)

demo:             ## build the demo store, then diagnose -> repair the regressed candidate
	./scripts/demo.sh

serve:            ## API on http://127.0.0.1:8000/docs
	$(PY) -m regressiontrace.cli serve

test:
	$(NICE) $(PY) -m pytest -q

lint:
	$(PY) -m ruff check regressiontrace benchmark tests scripts

bench-quick:      ## ~2.5 min smoke benchmark
	$(NICE) $(PY) -m benchmark.run --quick

bench:            ## full benchmark (~40 min on a 4-thread budget); results -> benchmark/results/
	$(NICE) $(PY) -m benchmark.run

clean:            ## delete generated state (store, MLflow runs); keeps the downloaded dataset
	rm -rf data/regressiontrace.duckdb mlflow.db mlartifacts .pytest_cache
