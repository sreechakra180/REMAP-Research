# =============================================================================
# REMAP-Net: Project Makefile
# =============================================================================
# Usage: make <target>
# =============================================================================

.PHONY: help train test lint format docker-build docker-run clean install dev-install \
        pre-commit check coverage tensorboard

# Default target
.DEFAULT_GOAL := help

# Project variables
PROJECT_NAME := remap-net
PACKAGE_NAME := remap_net
PYTHON := python
PIP := pip
DOCKER_IMAGE := remap-net:latest
DOCKER_RUNTIME := --gpus all

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help: ## Show this help message
	@echo "REMAP-Net Makefile"
	@echo "=================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install: ## Install the package
	$(PIP) install -e .

dev-install: ## Install with dev dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
train: ## Run training (use ARGS="..." for extra args)
	$(PYTHON) -m $(PACKAGE_NAME).cli.train $(ARGS)

eval: ## Run evaluation (use ARGS="..." for extra args)
	$(PYTHON) -m $(PACKAGE_NAME).cli.evaluate $(ARGS)

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: ## Run tests
	$(PYTHON) -m pytest tests/ -v --tb=short -ra

test-fast: ## Run tests excluding slow tests
	$(PYTHON) -m pytest tests/ -v --tb=short -ra -m "not slow"

coverage: ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint: ## Run all linters
	$(PYTHON) -m ruff check $(PACKAGE_NAME)/ tests/
	$(PYTHON) -m black --check $(PACKAGE_NAME)/ tests/
	$(PYTHON) -m isort --check-only $(PACKAGE_NAME)/ tests/

format: ## Auto-format code with black and isort
	$(PYTHON) -m isort $(PACKAGE_NAME)/ tests/
	$(PYTHON) -m black $(PACKAGE_NAME)/ tests/
	$(PYTHON) -m ruff check --fix $(PACKAGE_NAME)/ tests/

check: ## Run linters, type checks, and tests
	@echo "=== Formatting Check ==="
	$(PYTHON) -m black --check $(PACKAGE_NAME)/ tests/
	@echo "=== Import Sort Check ==="
	$(PYTHON) -m isort --check-only $(PACKAGE_NAME)/ tests/
	@echo "=== Linting ==="
	$(PYTHON) -m ruff check $(PACKAGE_NAME)/ tests/
	@echo "=== Tests ==="
	$(PYTHON) -m pytest tests/ -v --tb=short -ra
	@echo "=== All checks passed ==="

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE) .

docker-run: ## Run Docker container with GPU support
	docker run $(DOCKER_RUNTIME) -it \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/outputs:/workspace/outputs \
		-p 6006:6006 \
		-p 5000:5000 \
		$(DOCKER_IMAGE) $(CMD)

docker-shell: ## Open shell in Docker container
	docker run $(DOCKER_RUNTIME) -it \
		-v $(PWD)/data:/workspace/data \
		-v $(PWD)/outputs:/workspace/outputs \
		$(DOCKER_IMAGE) /bin/bash

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
tensorboard: ## Launch TensorBoard
	tensorboard --logdir=outputs/tensorboard --port=6006

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
	rm -rf htmlcov/ .coverage
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts and caches."

clean-outputs: ## Remove training outputs (use with caution!)
	@echo "WARNING: This will delete all training outputs!"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf outputs/
	@echo "Outputs cleaned."
