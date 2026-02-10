# POaaS: Minimal-Edit Prompt Optimization as a Service

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

POaaS is a lightweight prompt optimization system for small language models. It applies minimal edits through three CPU-routed specialist workers, using conservative skip logic and drift-controlled merging. Described in our FEVER 2026 submission.

## Replication Package

This repository provides everything needed to reproduce the manuscript experiments:

- POaaS system with CPU-only heuristic routing
- Baseline implementations: EvoPrompt, OPRO, PromptWizard, APO
- Evaluation framework with real model inference
- Automated dataset and model download scripts
- Docker and local service deployment
- Ablation support (`no_skip`, `no_drift` modes)

## System Overview

POaaS optimizes prompts through three specialist workers:

- **Cleaner**: Fixes typos, grammar, and spacing issues
- **Paraphraser**: Improves clarity and fluency
- **Fact-Adder**: Adds relevant contextual facts (at most 3 facts, at most 120 tokens)

## Architecture

```
User Query -> POaaS Orchestrator -> [Workers] -> Optimized Prompt
                     |
              Quality Analysis
              (typo, completeness, fluency, clarity)
                     |
              Conservative Skip Logic
              (skip if quality > 0.75 and typo < 0.20)
                     |
              Worker Selection & Parallel Processing
                     |
              Drift-Controlled Merging
              (delta <= 0.18, rho <= 2.4)
```

## Key Features

- **Conservative Skip Logic**: Skips optimization for high-quality prompts
- **Drift Control**: Bounds semantic drift (delta <= 0.18) and length expansion (rho <= 2.4)
- **CPU-Only Heuristics**: Quality scoring without GPU-dependent models
- **Parallel Processing**: Workers execute concurrently for low latency

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (A100 40GB+ recommended for full experiments)
- [vLLM](https://docs.vllm.ai/) for local model serving: `pip install vllm`
- Hugging Face account with [Llama model access](https://huggingface.co/meta-llama)

## Quick Start

1. **Install dependencies**:

```bash
pip install -r requirements.txt
cp .env.example .env   # Edit with your VLLM_URL, HF token, etc.
```

2. **Download models** (requires Hugging Face access to Llama):

```bash
bash scripts/download_models.sh
```

3. **Download evaluation datasets**:

```bash
python scripts/download_datasets.py
```

4. **Start POaaS services**:

```bash
python scripts/start_services.py
```

5. **Run experiments**:

```bash
# Quick test
python scripts/run_experiments.py --config test --limit 10

# Full manuscript experiments
python scripts/run_experiments.py --config manuscript_full
```

See [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for detailed replication instructions and [docs/paper_to_code_map.md](docs/paper_to_code_map.md) for a mapping of paper claims to code.

## System Validation

```bash
# Quick system check
python scripts/test_system.py --quick

# Full integration test
python scripts/test_system.py --full
```

## Configuration

System parameters are defined in `config/decoding.json`:

- **Quality Thresholds**: typo (0.30), completeness (0.70), fluency (0.80)
- **Skip Threshold**: 0.25 (conservative skip for high-quality prompts)
- **Drift Bounds**: max_drift (0.18), max_length_ratio (2.4)
- **Fact Constraints**: max_facts (3), fact_token_limit (120)

See `.env.example` for environment variable configuration.

## Evaluation

**Benchmarks:**
- **Reasoning**: BBH, GSM8K, CommonsenseQA
- **Factuality**: HaluEval, HalluLens, FActScore

**Baselines:**
- **EvoPrompt**: Evolutionary algorithm with LLM-guided mutations
- **OPRO**: Meta-prompting optimization with performance history
- **PromptWizard**: Task-aware critique and synthesis
- **APO**: Textual gradients with beam search

```bash
# Evaluate specific methods and benchmarks
python eval/real_evaluation.py --methods poaas evoprompt apo --benchmarks bbh --limit 10

# Full manuscript replication
python scripts/run_experiments.py --config manuscript_full
```

## Citation

<!-- TODO: Update citation fields (author, title, pages, DOI) when camera-ready
     is available. Manuscript title may differ from repository title. Verify
     venue (ACL vs FEVER 2026) against final acceptance notification. -->
```bibtex
@inproceedings{poaas2026,
  title={POaaS: Minimal-Edit Prompt Optimization as a Service},
  booktitle={FEVER 2026},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
