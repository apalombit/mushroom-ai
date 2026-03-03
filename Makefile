.PHONY: help setup up down lint test test-setup eval eval-benchmark

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

up:  ## Start all Docker services
	docker compose up -d

down:  ## Stop all Docker services
	docker compose down

test-setup:  ## Smoke test: verify all services connected
	python -m scripts.test_setup

lint:  ## Run linter
	ruff check .
	ruff format --check .

format:  ## Auto-format code
	ruff check --fix .
	ruff format .

test:  ## Run tests
	pytest -v

db-init:  ## Create database tables
	python -m scripts.init_db

ingest:  ## Run ingestion pipeline
	python -m scripts.ingest

serve:  ## Start FastAPI server
	uvicorn api.main:app --reload --port 8001

ui:  ## Start Streamlit dashboard
	streamlit run ui/app.py --server.port 8501

eval:  ## Evaluate retrieval recall (logs to MLflow)
	python -m scripts.eval

eval-benchmark:  ## Run benchmark across weight configs
	python -m scripts.eval --benchmark
