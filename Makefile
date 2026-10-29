PYTHON ?= python
CDD ?= cdd

.PHONY: help install lint format typecheck test cov check data pilot clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install the package with development tools
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:  ## Lint and check formatting
	ruff check src tests examples
	ruff format --check src tests examples

format:  ## Apply formatting and safe lint fixes
	ruff format src tests examples
	ruff check --fix src tests examples

typecheck:  ## Type-check the package (mypy --strict)
	mypy

test:  ## Run the test suite
	pytest

cov:  ## Run the tests with a coverage report
	pytest --cov --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs

data:  ## Generate the pilot benchmark streams
	$(CDD) generate --config configs/benchmark/pilot.yaml --out data/pilot

pilot: data  ## Generate the pilot streams and run the pilot experiment
	$(CDD) -v run --config configs/experiments/pilot.yaml

clean:  ## Remove caches and build output (keeps data/ and runs/)
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
