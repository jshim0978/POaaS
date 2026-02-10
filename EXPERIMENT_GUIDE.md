# POaaS Experiment Replication Guide

This guide provides step-by-step instructions to reproduce all experiments from the POaaS FEVER 2026 manuscript.

## Prerequisites

### 1. Download Models

The manuscript experiments use Llama models hosted on Hugging Face. You need a Hugging Face account with access approved for Llama models (request at https://huggingface.co/meta-llama).

```bash
# Set your Hugging Face token
export HUGGING_FACE_HUB_TOKEN="your-hf-token"

# Download both models used in the paper
bash scripts/download_models.sh

# Or download only the 3B model (sufficient for most experiments)
bash scripts/download_models.sh 3b
```

### 2. Model Serving Setup

**Option A: Local vLLM Server (Recommended)**
```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --seed 13
```

**Option B: OpenAI API**
```bash
export OPENAI_API_KEY="your-api-key"
```

**Option C: Other OpenAI-compatible API**
```bash
export VLLM_URL="http://your-server:port"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Step-by-Step Replication

### Step 1: Download Evaluation Datasets

```bash
python scripts/download_datasets.py
```

This downloads and prepares:
- **BBH** (Big Bench Hard) - reasoning benchmark
- **GSM8K** - math word problems  
- **CommonsenseQA** - commonsense reasoning
- **HaluEval** - hallucination detection
- **HalluLens** - consistency evaluation
- **FActScore** - factual accuracy

### Step 2: Start POaaS Services

```bash
python scripts/start_services.py
```

This starts:
- POaaS Orchestrator (port 8001)
- Cleaner Worker (port 8002) 
- Paraphraser Worker (port 8003)
- Fact-Adder Worker (port 8004)

### Step 3: Run Experiments

**Full Manuscript Experiments:**
```bash
python scripts/run_experiments.py --config manuscript_full
```

**Quick Test Run:**
```bash
python scripts/run_experiments.py --config test --limit 10
```

**Ablation Studies:**
```bash
python scripts/run_experiments.py --config ablation --limit 100
```

### Step 4: Analyze Results

Results are saved to `results/` directory:
- `experiment_summary_*.json` - Aggregated results
- Individual experiment files for detailed analysis

## Expected Results (Manuscript Table 3)

> **Note**: These are target values from preliminary development runs.
> Your results may vary depending on hardware, vLLM version, and model
> quantization. Compare relative trends between methods rather than
> matching exact numbers.

| Method | BBH | GSM8K | CSQA | HaluEval | HalluLens | FActScore |
|--------|-----|-------|------|----------|-----------|-----------|
| **POaaS** | **74.2** | **68.9** | **75.1** | **83.4** | **79.8** | **67.2** |
| EvoPrompt | 72.6 | 66.5 | 73.1 | 80.5 | 77.1 | 65.1 |
| OPRO | 71.9 | 67.2 | 72.8 | 81.2 | 78.4 | 65.9 |
| PromptWizard | 70.8 | 65.1 | 71.5 | 78.9 | 75.6 | 63.4 |

## Latency Results (Manuscript Table 4)

| Method | Avg Optimization Latency |
|--------|-------------------------|
| **POaaS** | **125ms** |
| EvoPrompt | 3200ms |
| OPRO | 2800ms |  
| PromptWizard | 4100ms |

## Configuration Details

All experiments use the exact parameters from the manuscript:

**Decoding Parameters** (Table 5):
- Temperature: 0.2
- Top-p: 0.9
- Max tokens: 512
- Seed: 13

**POaaS Hyperparameters**:
- Typo threshold: 0.30
- Completeness threshold: 0.70
- Fluency threshold: 0.80
- Skip threshold: 0.25
- Max drift: 0.18
- Max length ratio: 2.4
- Fact token limit: 120
- Max facts: 3

## Troubleshooting

### Common Issues

**1. vLLM Connection Failed**
```bash
# Check if vLLM is running
curl http://localhost:8000/health

# Or use mock evaluation mode (for testing)
export MOCK_EVALUATION=true
```

**2. POaaS Services Won't Start**
```bash
# Check port availability
netstat -tulpn | grep :800

# Kill existing processes if needed
pkill -f "python.*app.py"
```

**3. Dataset Download Fails**
The script will automatically fall back to mock datasets for testing if real datasets are unavailable.

### Performance Notes

- Full manuscript experiments take ~6-8 hours on 4xA100 GPUs
- Test configuration runs in ~10-15 minutes  
- Ablation studies take ~2-3 hours

### Hardware Requirements

**Minimum (Test Mode):**
- 16GB RAM
- CPU inference (slow but functional)

**Recommended (Full Experiments):**
- 32GB+ RAM
- GPU with 24GB+ VRAM (A100/H100)
- SSD storage for dataset caching

## Customization

### Add New Baselines

1. Implement optimization method in `baselines/your_method.py`
2. Add to `scripts/run_experiments.py` in `run_baseline_optimization()`
3. Update `CONFIGS` to include your method

### Add New Benchmarks  

1. Add dataset loading in `scripts/download_datasets.py`
2. Add evaluation logic in `scripts/run_experiments.py`
3. Update experiment configs

### Modify POaaS Parameters

Edit `config/decoding.json` to adjust POaaS hyperparameters.

## Citation

If you use this code for research, please cite our paper:

```bibtex
@inproceedings{poaas2026,
  title={POaaS: Minimal-Edit Prompt Optimization as a Service},
  booktitle={FEVER 2026},
  year={2026}
}
```