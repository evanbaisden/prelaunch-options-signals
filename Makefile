.PHONY: help setup lint format test run clean

help:
	@echo "Available commands:"
	@echo "  setup   - Install dependencies from requirements.txt"
	@echo "  lint    - Run linting with ruff and black check"
	@echo "  format  - Format code with black"
	@echo "  test    - Run tests with pytest"
	@echo "  run     - Run full analysis (all events)"
	@echo "  clean   - Clean up cache and temporary files"

setup:
	pip install -r requirements.txt
	pip install ruff black pytest

lint:
	ruff check .
	black --check .

format:
	black .

test:
	pytest -q

run:
	python -m src.analysis run-all

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +