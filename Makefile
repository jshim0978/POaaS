# POaaS Makefile
# Convenience commands for development and reproduction

.PHONY: help install test clean docker run-services download reproduce

# Default target
help:
	@echo "POaaS - Minimal-Edit Prompt Optimization as a Service"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make download-models- Download Llama models from HF"
	@echo "  make download       - Download evaluation datasets"
	@echo "  make download-all   - Download models + datasets"
	@echo "  make test           - Run quick system test"
	@echo "  make run-services   - Start all POaaS services"
	@echo "  make reproduce      - Run full manuscript experiments"
	@echo "  make reproduce-test - Run quick reproduction test"
	@echo "  make docker         - Build Docker image"
	@echo "  make docker-up      - Start services with Docker Compose"
	@echo "  make clean          - Clean generated files"
	@echo "  make aggregate      - Aggregate results into tables"
	@echo ""

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

# Download models from Hugging Face (requires HF token)
download-models:
	bash scripts/download_models.sh

# Download evaluation datasets
download:
	python scripts/download_datasets.py --seed 13 --samples 500

# Download everything (models + datasets)
download-all: download-models download

# Run system tests
test:
	python scripts/test_system.py --quick

test-full:
	python scripts/test_system.py --full

# Start services
run-services:
	python scripts/start_services.py

# Docker commands
docker:
	docker build -t poaas:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Reproduction commands
reproduce-test:
	python scripts/run_experiments.py --config test --limit 10

reproduce:
	python scripts/run_experiments.py --config manuscript_full

reproduce-ablation:
	python scripts/run_experiments.py --config ablation

# Aggregate results
aggregate:
	python scripts/aggregate_results.py

# Evaluation shortcuts
eval-bbh:
	python3 eval/real_evaluation.py --benchmarks bbh --methods poaas evoprompt --limit 50

eval-gsm8k:
	python3 eval/real_evaluation.py --benchmarks gsm8k --methods poaas evoprompt --limit 50

eval-all:
	python3 eval/real_evaluation.py \
		--benchmarks bbh gsm8k commonsenseqa halueval hallulens factscore \
		--methods poaas evoprompt opro promptwizard

# Full experiments with all noise conditions
eval-full:
	python3 scripts/run_full_experiments.py --all --limit 500

eval-full-test:
	python3 scripts/run_full_experiments.py --benchmarks bbh gsm8k --methods poaas baseline --limit 10

# Ablation experiments
ablation-no-skip:
	POAAS_ABLATION=no_skip python eval/real_evaluation.py \
		--benchmarks bbh gsm8k commonsenseqa --methods poaas --limit 100

ablation-no-drift:
	POAAS_ABLATION=no_drift python eval/real_evaluation.py \
		--benchmarks bbh gsm8k commonsenseqa --methods poaas --limit 100

# Clean generated files
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.pyc */*.pyc */*/*.pyc
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf build dist

clean-results:
	rm -rf results/*.json results/*.log

clean-runs:
	rm -rf runs/*

clean-all: clean clean-results clean-runs

# Linting and formatting
lint:
	ruff check .
	
format:
	black .

# Health checks
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8001/health | python -m json.tool || echo "Orchestrator: DOWN"
	@curl -s http://localhost:8002/health | python -m json.tool || echo "Cleaner: DOWN"
	@curl -s http://localhost:8003/health | python -m json.tool || echo "Paraphraser: DOWN"
	@curl -s http://localhost:8004/health | python -m json.tool || echo "Fact-Adder: DOWN"

# Quick demo
demo:
	@echo "Running POaaS demo..."
	@curl -s -X POST http://localhost:8001/infer \
		-H "Content-Type: application/json" \
		-d '{"prompt": "waht is the captial of frnace?"}' | python -m json.tool

