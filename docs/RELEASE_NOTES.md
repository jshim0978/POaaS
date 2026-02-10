# POaaS v1.0 -- FEVER 2026 Submission

Prompt Optimization as a Service: minimal-edit prompt refinement through three
CPU-routed specialists (Cleaner, Paraphraser, Fact-Adder) with conservative
skip logic and drift-controlled merging.

## Contents

- Full POaaS system implementation (orchestrator + 3 specialist workers)
- Four baseline methods: EvoPrompt, OPRO, PromptWizard, APO
- Evaluation framework for 6 benchmarks:
  - Reasoning: BBH, GSM8K, CommonsenseQA
  - Factuality: HaluEval, HalluLens, FActScore
- Automated dataset download (`scripts/download_datasets.py`)
- Automated model download (`scripts/download_models.sh`)
- Docker deployment configuration
- Ablation support (`no_skip`, `no_drift` modes)
- Noise injection for degradation robustness experiments (5%, 10%, 15%)

## Requirements

- Python 3.8+
- vLLM or OpenAI API access
- Llama-3.2-3B-Instruct (primary) / Llama-3.1-8B-Instruct (cross-model)
- Hugging Face account with Llama access approved

## Quick Start

```bash
pip install -r requirements.txt
bash scripts/download_models.sh
python scripts/download_datasets.py
python scripts/start_services.py
python scripts/run_experiments.py --config test --limit 10
```

## Documentation

- `README.md` -- System overview and quick start
- `EXPERIMENT_GUIDE.md` -- Step-by-step experiment replication
- `REPRODUCIBILITY.md` -- Full reproducibility instructions
- `docs/paper_to_code_map.md` -- Paper claims mapped to code
- `docs/MIGRATION_NOTES.md` -- MPR-SaaS to POaaS history
