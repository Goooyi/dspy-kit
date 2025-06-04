# Makefile for dspy-kit development

.PHONY: install test lint format clean build publish help

# Development setup
install:
	uv sync --dev
	uv run pre-commit install

# Testing
test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ --cov=dspy_evals --cov-report=html --cov-report=term

test-integration:
	uv run pytest tests/integration/ -v -m integration

# Code quality
lint:
	uv run ruff check dspy_evals tests
	uv run mypy dspy_evals

format:
	uv run ruff format dspy_evals tests examples
	uv run ruff check --fix dspy_evals tests

# Documentation
docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

# Building and publishing
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

publish-test: build
	uv publish --repository testpypi

publish: build
	uv publish

# Development helpers
dev-setup: install
	@echo "Setting up development environment..."
	@echo "Run 'source .venv/bin/activate' to activate the environment"

quick-test:
	uv run pytest tests/unit/ -x -v

# Integration with your DSPy project
link-to-tracer:
	@echo "Creating editable install in your DSPy project..."
	@read -p "Enter path to your DSPy project: " PROJECT_PATH && \
	cd $$PROJECT_PATH && \
	uv add --editable $(PWD)

help:
	@echo "Available commands:"
	@echo "  install          Install dependencies and pre-commit hooks"
	@echo "  test            Run all tests"
	@echo "  test-cov        Run tests with coverage"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code"
	@echo "  build           Build the package"
	@echo "  publish-test    Publish to test PyPI"
	@echo "  publish         Publish to PyPI"
	@echo "  clean           Clean build artifacts"
	@echo "  dev-setup       Complete development setup"
	@echo "  link-to-tracer  Link to your DSPy project"
