PY      ?= python
CONFIG  ?= configs/qlora-3b.yaml
T4CONFIG ?= configs/qlora-3b-t4.yaml

.PHONY: install install-colab install-fast data smoke smoke-t4 train train-t4 eval export info test lint format lock clean

install:
	$(PY) -m pip install -e ".[train]"

# Colab T4: keep the pre-installed torch; install everything else.
install-colab:
	$(PY) -c "import torch; print(f'torch=={torch.__version__}')" > /tmp/colab-constraints.txt
	$(PY) -m pip install -e ".[colab]" -c /tmp/colab-constraints.txt

install-fast:
	$(PY) -m pip install -r requirements-fast.txt --no-build-isolation

data:
	$(PY) -m cyberft.cli data --config $(CONFIG)

# 400 rows -> 5 training steps. Catches dtype, collator and label problems in ~2 min.
smoke:
	$(PY) -m cyberft.cli data --config $(CONFIG) --smoke
	$(PY) -m cyberft.cli train --config $(CONFIG) --smoke

# Same gate on the T4 profile (1024 max_len, reduced caps).
smoke-t4:
	$(PY) -m cyberft.cli data --config $(T4CONFIG) --smoke
	$(PY) -m cyberft.cli train --config $(T4CONFIG) --smoke

train:
	$(PY) -m cyberft.cli train --config $(CONFIG)

train-t4:
	$(PY) -m cyberft.cli train --config $(T4CONFIG)

eval:
	$(PY) -m cyberft.cli eval --config $(CONFIG)

export:
	$(PY) -m cyberft.cli export --config $(CONFIG)

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
