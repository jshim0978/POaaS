# POaaS Reproducibility Guide

This document provides step-by-step instructions to reproduce all experiments from the POaaS FEVER 2026 paper.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Running Experiments](#running-experiments)
5. [Reproducing Paper Tables](#reproducing-paper-tables)
6. [Ablation Studies](#ablation-studies)
7. [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Minimum (Test Mode)
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Storage**: 10GB free space
- **GPU**: Not required (uses API fallback)

### Recommended (Full Experiments)
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **GPU**: NVIDIA A100 (40GB) or H100 (80GB)
- **Storage**: 50GB+ SSD
- **OS**: Ubuntu 20.04+ or similar Linux

### Paper Experiments
- **Hardware**: 4x NVIDIA A100 80GB GPUs
- **Runtime**: ~6-8 hours for full manuscript experiments
- **VRAM**: 24GB minimum per model

---

## Environment Setup

### Option A: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/jshim0978/POaaS.git
cd POaaS

# Build and start services
docker-compose up -d

# Verify services are running
curl http://localhost:8001/health
```

### Option B: Conda Environment

```bash
# Create conda environment
conda create -n poaas python=3.10 -y
conda activate poaas

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Start services
python scripts/start_services.py
```

### Option C: vLLM with GPU

```bash
# Install vLLM
pip install vllm

# Start vLLM server with target model
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --seed 13

# In another terminal, start POaaS services
export VLLM_URL=http://localhost:8000
python scripts/start_services.py
```

---

## Dataset Preparation

### Download All Datasets

```bash
python scripts/download_datasets.py --seed 13 --samples 500
```

This downloads and prepares:
- **BBH** (Big Bench Hard) - 23 tasks with stratified sampling
- **GSM8K** - Math word problems (test split)
- **CommonsenseQA** - Multiple choice reasoning (dev split)
- **HaluEval** - Hallucination detection
- **HalluLens** - Consistency evaluation
- **FActScore** - Factual accuracy (biography generation)

### Verify Datasets

```bash
# Check dataset files
ls -la eval/data/

# Expected output:
# bbh.jsonl           (500 samples, stratified across tasks)
# gsm8k.jsonl         (500 samples)
# commonsenseqa.jsonl (500 samples)
# halueval.jsonl      (500 samples)
# hallulens.jsonl     (500 samples)
# factscore.jsonl     (500 samples)

# Check sample indices (for reproducibility)
ls -la sample_indices/
```

---

## Running Experiments

### Quick Test (Sanity Check)

```bash
# Run minimal test to verify setup
python scripts/run_experiments.py --config test --limit 10

# Expected runtime: ~5-10 minutes
# Expected output: Results in results/ directory
```

### Full Manuscript Experiments

```bash
# Run all experiments from the paper
python scripts/run_experiments.py --config manuscript_full

# Expected runtime: 6-8 hours on 4xA100
# Results saved to: results/
```

### Specific Benchmark

```bash
# Run specific benchmark with specific methods
python eval/real_evaluation.py \
    --benchmarks bbh gsm8k \
    --methods poaas evoprompt opro \
    --limit 100
```

---

## Reproducing Paper Tables

### Table 3: Main Results (Clean Inputs)

```bash
# Run main experiments
python scripts/run_experiments.py --config manuscript_full

# Results will be in:
# - results/real_eval_*_summary.json (aggregated)
# - results/real_eval_*_<method>_<benchmark>.json (detailed)
```

Expected results (Llama-3.2-3B-Instruct):

> **Note**: These are target values from preliminary development runs.
> Your results may differ due to hardware, vLLM version, or model quantization.
> Run the full evaluation pipeline and compare trends rather than exact numbers.

| Method | BBH | GSM8K | CSQA | HaluEval | HalluLens | FActScore |
|--------|-----|-------|------|----------|-----------|-----------|
| POaaS | 74.2 | 68.9 | 75.1 | 83.4 | 79.8 | 67.2 |
| EvoPrompt | 72.6 | 66.5 | 73.1 | 80.5 | 77.1 | 65.1 |
| OPRO | 71.9 | 67.2 | 72.8 | 81.2 | 78.4 | 65.9 |
| PromptWizard | 70.8 | 65.1 | 71.5 | 78.9 | 75.6 | 63.4 |

### Table 4: Latency Comparison

```bash
# Latency is logged in results files
# Extract with:
python -c "
import json
from pathlib import Path

for f in Path('results').glob('*summary.json'):
    with open(f) as fp:
        data = json.load(fp)
    for bench, methods in data.items():
        for method, results in methods.items():
            lat = results.get('avg_optimization_latency_ms', 0)
            print(f'{method:15} {bench:15} {lat:.1f}ms')
"
```

### Table 5: Degraded Inputs

```bash
# Run with noise injection
python eval/real_evaluation.py \
    --benchmarks bbh gsm8k \
    --methods poaas evoprompt \
    --noise-type deletion \
    --noise-rate 0.10 \
    --limit 500
```

### Table 6: Ablation Study

```bash
# Run ablation experiments
python scripts/run_experiments.py --config ablation

# Or run specific ablations:
POAAS_ABLATION=no_skip python eval/real_evaluation.py --benchmarks bbh --methods poaas
POAAS_ABLATION=no_drift python eval/real_evaluation.py --benchmarks bbh --methods poaas
```

---

## Ablation Studies

### Skip Logic Ablation

```bash
# Disable skip logic
export POAAS_ABLATION=no_skip
python eval/real_evaluation.py --benchmarks bbh gsm8k commonsenseqa --methods poaas

# Or via API
curl -X POST http://localhost:8001/infer \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is 2+2?", "ablation": "no_skip"}'
```

### Drift Control Ablation

```bash
# Disable drift control
export POAAS_ABLATION=no_drift
python eval/real_evaluation.py --benchmarks bbh gsm8k commonsenseqa --methods poaas
```

---

## Configuration Reference

### Key Configuration Files

- `config/decoding.json` - Model and decoding parameters
- `config/prices.yml` - Cost computation settings

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_URL` | `http://localhost:8000` | vLLM server URL |
| `HUGGING_FACE_HUB_TOKEN` | (none) | Hugging Face token (required for gated Llama models) |
| `OPENAI_API_KEY` | (none) | OpenAI API key (only if using GPT models instead of vLLM) |
| `POAAS_ABLATION` | `full` | Ablation mode: full, no_skip, no_drift |
| `POAAS_RUNS_DIR` | `./runs` | Directory for run artifacts |
| `CLEANER_URL` | `http://localhost:8002` | Cleaner worker URL (only for remote deployment) |
| `PARAPHRASER_URL` | `http://localhost:8003` | Paraphraser worker URL (only for remote deployment) |
| `FACT_ADDER_URL` | `http://localhost:8004` | Fact-Adder worker URL (only for remote deployment) |

### Decoding Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.2 |
| Top-p | 0.9 |
| Max tokens | 512 |
| Seed | 13 |

### POaaS Thresholds

| Parameter | Value | Paper Reference |
|-----------|-------|-----------------|
| Typo threshold | 0.30 | Section 3.2 |
| Completeness threshold | 0.70 | Section 3.2 |
| Fluency threshold | 0.80 | Section 3.2 |
| Skip threshold | 0.25 | Section 3.2 |
| Max drift (δ) | 0.18 | Section 3.5 |
| Max length ratio (ρ) | 2.4 | Section 3.5 |
| Max facts | 3 | Section 3.1 |
| Fact token limit | 120 | Section 3.1 |

---

## Troubleshooting

### Common Issues

#### 1. vLLM Connection Failed

```bash
# Check if vLLM is running
curl http://localhost:8000/health

# Use mock evaluation mode for testing
export MOCK_EVALUATION=true
python eval/real_evaluation.py --limit 10
```

#### 2. CUDA Out of Memory

```bash
# Use smaller batch size or model
# For 8B model, ensure 24GB+ VRAM

# Or use OpenAI API instead
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-3.5-turbo"
```

#### 3. Dataset Download Fails

```bash
# Datasets will fallback to mock data
# Check network connectivity
# Manual download instructions in eval/data/README.md
```

#### 4. Results Don't Match Paper

- Ensure seed=13 is set consistently
- Verify sample indices match (check sample_indices/)
- Check model version matches (Llama-3.2-3B-Instruct)
- Verify decoding parameters match config/decoding.json

### Getting Help

1. Check logs: `tail -f results/evaluation.log`
2. Run system test: `python scripts/test_system.py --full`
3. Open an issue with:
   - Environment details
   - Full error message
   - Steps to reproduce

---

## Verification Checklist

Before reporting results, verify:

- [ ] All 6 datasets downloaded (500 samples each)
- [ ] Sample indices match those in sample_indices/
- [ ] Decoding parameters match Table 5
- [ ] Seed = 13 throughout
- [ ] Model = meta-llama/Llama-3.2-3B-Instruct (or 8B)
- [ ] Results within expected variance (±2% of paper)

---

## Citation

If you use this code for research, please cite:

```bibtex
@inproceedings{poaas2026,
  title={POaaS: Minimal-Edit Prompt Optimization as a Service},
  booktitle={FEVER 2026},
  year={2026}
}
```

