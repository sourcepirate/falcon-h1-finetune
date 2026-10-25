PY      ?= python
CONFIG  ?= configs/qlora-3b.yaml
ADAPTER ?=

.PHONY: install install-fast data smoke train eval export info test lint format lock clean

install:
	$(PY) -m pip install -e ".[train]"

install-fast:
	$(PY) -m pip install -r requirements-fast.txt --no-build-isolation

data:
	$(PY) -m cyberft.cli data --config $(CONFIG)

# 400 rows -> 5 training steps. Catches dtype, collator and label problems in ~2 min.
smoke:
	$(PY) -m cyberft.cli data --config $(CONFIG) --smoke
	$(PY) -m cyberft.cli train --config $(CONFIG) --smoke

train:
	$(PY) -m cyberft.cli train --config $(CONFIG)

eval:
	$(PY) -m cyberft.cli eval --config $(CONFIG) $(if $(ADAPTER),--adapter $(ADAPTER))

export:
	$(PY) -m cyberft.cli export --config $(CONFIG) $(if $(ADAPTER),--adapter $(ADAPTER))

info:
	$(PY) -m cyberft.cli info --config $(CONFIG)

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m mypy

format:
	$(PY) -m ruff format .

lock:
	$(PY) -m pip freeze > requirements.lock

clean:
	rm -rf outputs/smoke data/processed/smoke .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
