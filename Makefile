# Prelaunch Options Signals - Phase 1 Analysis Makefile
# Provides convenient automation commands for development and analysis

# Configuration
PYTHON := python
VENV_DIR := .venv
REQUIREMENTS := requirements.txt
DATA_DIR := data/raw
RESULTS_DIR := results
NOTEBOOKS_DIR := notebooks

# Default environment file
ENV_FILE := .env

.PHONY: help install setup clean test lint format run-phase1 validate-data check-env serve-notebook

# Default target
help: ## Display available commands
	@echo "Prelaunch Options Signals - Phase 1 Analysis"
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment: $(ENV_FILE)"
	@echo "Data directory: $(DATA_DIR)"
	@echo "Results directory: $(RESULTS_DIR)"

install: ## Install project dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQUIREMENTS)
	@echo "✅ Dependencies installed successfully"

setup: ## Set up development environment
	@echo "Setting up development environment..."
	@if not exist "$(ENV_FILE)" copy ".env.example" "$(ENV_FILE)"
	@if not exist "$(DATA_DIR)" mkdir "$(DATA_DIR)"
	@if not exist "$(RESULTS_DIR)" mkdir "$(RESULTS_DIR)"
	@if not exist "$(NOTEBOOKS_DIR)" mkdir "$(NOTEBOOKS_DIR)"
	@echo "✅ Development environment ready"
	@echo "📝 Please review $(ENV_FILE) and update configuration as needed"

clean: ## Clean temporary files and caches
	@echo "Cleaning temporary files..."
	@if exist "__pycache__" rmdir /s /q "__pycache__"
	@if exist ".pytest_cache" rmdir /s /q ".pytest_cache"
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
	@for /d /r . %%d in (*.egg-info) do @if exist "%%d" rmdir /s /q "%%d"
	@echo "✅ Cleanup complete"

test: ## Run test suite
	@echo "Running test suite..."
	$(PYTHON) -m pytest tests/ -v
	@echo "✅ Tests completed"

lint: ## Run code linting
	@echo "Running code analysis..."
	$(PYTHON) -m flake8 src/ --max-line-length=100 --ignore=E501,W503
	@echo "✅ Linting complete"

format: ## Format code using black
	@echo "Formatting code..."
	$(PYTHON) -m black src/ tests/ --line-length=100
	@echo "✅ Code formatting complete"

validate-data: ## Validate required data files exist
	@echo "Validating data files..."
	$(PYTHON) scripts/run_phase1.py --dry-run
	@echo "✅ Data validation complete"

check-env: ## Check environment configuration
	@echo "Environment Configuration Check:"
	@echo "================================"
	@if exist "$(ENV_FILE)" (echo "✅ Configuration file: $(ENV_FILE)" && type "$(ENV_FILE)") else echo "❌ Configuration file missing: $(ENV_FILE)"
	@echo ""
	@echo "Required directories:"
	@if exist "$(DATA_DIR)" (echo "✅ Data directory: $(DATA_DIR)") else echo "❌ Data directory missing: $(DATA_DIR)"
	@if exist "$(RESULTS_DIR)" (echo "✅ Results directory: $(RESULTS_DIR)") else echo "❌ Results directory missing: $(RESULTS_DIR)"

run-phase1: ## Run complete Phase 1 analysis
	@echo "🚀 Starting Phase 1 Analysis..."
	$(PYTHON) scripts/run_phase1.py
	@echo "✅ Phase 1 analysis complete"

run-phase1-verbose: ## Run Phase 1 analysis with verbose output
	@echo "🚀 Starting Phase 1 Analysis (verbose mode)..."
	$(PYTHON) scripts/run_phase1.py --verbose
	@echo "✅ Phase 1 analysis complete"

run-phase1-quiet: ## Run Phase 1 analysis with minimal output
	@echo "🚀 Starting Phase 1 Analysis (quiet mode)..."
	$(PYTHON) scripts/run_phase1.py --quiet
	@echo "✅ Phase 1 analysis complete"

serve-notebook: ## Start Jupyter notebook server
	@echo "Starting Jupyter notebook server..."
	$(PYTHON) -m jupyter notebook --notebook-dir=$(NOTEBOOKS_DIR) --ip=127.0.0.1 --port=8888

# Development workflow commands
dev-setup: setup install check-env ## Complete development setup
	@echo "🎉 Development environment fully configured!"

dev-clean: clean ## Development cleanup
	@echo "🧹 Development cleanup complete"

# Quality assurance workflow
qa: lint test ## Run quality assurance checks
	@echo "🔍 Quality assurance checks complete"

# Full analysis workflow
analyze: validate-data run-phase1 ## Validate data and run complete analysis
	@echo "📊 Complete analysis workflow finished"

# Pre-commit workflow  
pre-commit: format lint test ## Run pre-commit checks
	@echo "✅ Pre-commit checks passed"

# Documentation generation (placeholder for future)
docs: ## Generate documentation (placeholder)
	@echo "📚 Documentation generation not yet implemented"

# Quick start for new users
quickstart: dev-setup validate-data ## Quick start for new users
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Add your stock data CSV files to $(DATA_DIR)/"
	@echo "2. Run 'make run-phase1' to execute the analysis"
	@echo "3. Check results in $(RESULTS_DIR)/"
	@echo ""
	@echo "For help: make help"

# Version and info
info: ## Display project information
	@echo "Prelaunch Options Signals Analysis - Phase 1"
	@echo "============================================"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Project root: $$(pwd)"
	@echo "Data directory: $(DATA_DIR)"
	@echo "Results directory: $(RESULTS_DIR)"
	@echo "Configuration: $(ENV_FILE)"
	@echo ""
	@echo "Key files:"
	@echo "- Analysis engine: src/phase1/run.py"
	@echo "- Configuration: src/config.py"
	@echo "- Main script: scripts/run_phase1.py"