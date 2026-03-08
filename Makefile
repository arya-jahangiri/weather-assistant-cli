PYTHON := $(shell if command -v python3 >/dev/null 2>&1; then printf 'python3'; elif command -v python >/dev/null 2>&1; then printf 'python'; elif command -v py >/dev/null 2>&1; then printf 'py -3'; fi)
VENV_PYTHON := $(firstword $(wildcard .venv/bin/python .venv/Scripts/python.exe))

.PHONY: setup run clean-reset format lint typecheck test check

define require-python
	@if [ -z "$(strip $(PYTHON))" ]; then echo "Python 3 launcher not found. Install Python 3.11+ and retry."; exit 1; fi
endef

define require-venv
	@if [ -z "$(strip $(VENV_PYTHON))" ]; then echo ".venv not found. Run 'make setup' first."; exit 1; fi
endef

setup:
	$(require-python)
	$(PYTHON) scripts/setup_dev.py

run:
	$(require-venv)
	$(VENV_PYTHON) main.py

clean-reset:
	$(require-python)
	$(PYTHON) scripts/clean_reset.py

format:
	$(require-venv)
	$(VENV_PYTHON) -m ruff format .

lint:
	$(require-venv)
	$(VENV_PYTHON) -m ruff check .

typecheck:
	$(require-venv)
	$(VENV_PYTHON) -m pyright

test:
	$(require-venv)
	$(VENV_PYTHON) -m pytest -q

check: format lint typecheck test
