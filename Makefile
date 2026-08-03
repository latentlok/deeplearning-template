.PHONY: help setup test lint fmt smoke train tb runs clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## install deps into .venv
	uv sync --extra dev

test:  ## run the test suite
	uv run pytest tests/ -q

lint:  ## check style
	uv run ruff check src tests scripts

fmt:  ## fix style
	uv run ruff format src tests scripts && uv run ruff check --fix src tests scripts

smoke:  ## 20-step end-to-end run; proves the wiring
	uv run python -m dlt.train experiment=e0

train:  ## EXP=<name> (default e0)
	uv run python -m dlt.train experiment=$(or $(EXP),e0)

tb:  ## tensorboard over every run
	./scripts/tb.sh

runs:  ## table of all runs
	uv run python scripts/runs.py

clean:  ## delete run outputs and caches
	rm -rf outputs multirun .pytest_cache .ruff_cache
