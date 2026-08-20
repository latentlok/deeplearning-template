.PHONY: help setup test lint fmt smoke train tb runs stats graph clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## install deps into .venv
	uv sync --extra dev

test:  ## run the test suite
	uv run pytest tests/ -q

lint:  ## check style
	uv run ruff check engine models dataset utils tests train.py eval.py

fmt:  ## fix style
	uv run ruff format engine models dataset utils tests train.py eval.py && \
	uv run ruff check --fix engine models dataset utils tests train.py eval.py

smoke:  ## 20-step end-to-end run; proves the wiring
	uv run python train.py experiment=e0

train:  ## EXP=<name> (default e0)
	uv run python train.py experiment=$(or $(EXP),e0)

stats:  ## scan $DL_DATA/train.zarr and write stats.json
	uv run python utils/stats.py

tb:  ## tensorboard over every run
	uv run tensorboard --logdir $(or $(LOGDIR),outputs) --port $(or $(PORT),6006)

runs:  ## table of all runs
	uv run python utils/runs.py

graph:  ## refresh the code graph into .graphify/ (structural, no LLM, nothing leaves the machine)
	graphify update . --no-cluster
	@mkdir -p .graphify && mv -f graphify-out/graph.json graphify-out/manifest.json .graphify/ && rm -rf graphify-out
	@echo "graph -> .graphify/graph.json  (read it with: graphify explain X --graph .graphify/graph.json)"

clean:  ## delete run outputs and caches
	rm -rf outputs multirun .pytest_cache .ruff_cache
