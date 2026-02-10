# Paper-to-Code Map

This document maps claims and components from the POaaS FEVER 2026 paper to
their concrete implementations in this repository.

## System Architecture (Section 3.1)

| Paper Component | Code Location | Description |
|-----------------|---------------|-------------|
| Three specialists | `workers/cleaner/app.py`, `workers/paraphraser/app.py`, `workers/fact_adder/app.py` | FastAPI services for each specialist |
| Orchestrator | `orchestrator/app.py` | Routing, skip logic, merging |
| Agent prompts | `prompts/cleaner.md`, `prompts/paraphraser.md`, `prompts/fact_adder.md` | System prompts for each specialist |
| Configuration | `config/decoding.json` | All hyperparameters (thresholds, decoding, ablations) |

## Quality Scoring and Routing (Section 3.2)

| Paper Claim | Code Location | Verification |
|-------------|---------------|--------------|
| CPU-only heuristic scoring (typo, completeness, fluency, clarity) | `orchestrator/app.py`: `compute_typo_score()`, `compute_completeness_score()`, `compute_fluency_score()`, `compute_clarity_score()` | `curl -X POST localhost:8001/infer -H 'Content-Type: application/json' -d '{"prompt":"test prompt"}'` |
| Overall quality score (Eq. 3) | `orchestrator/app.py`: `compute_quality_score()` | Returns component scores in response |
| Conservative skip logic | `orchestrator/app.py`: `should_skip()` | Compare `POAAS_ABLATION=full` vs `POAAS_ABLATION=no_skip` |
| Threshold-based worker selection | `orchestrator/app.py`: `select_workers()` | Workers chosen based on per-dimension thresholds |

## Drift-Controlled Merging (Section 3.5)

| Paper Claim | Code Location | Verification |
|-------------|---------------|--------------|
| Lexical similarity drift (Eq. 1) | `orchestrator/app.py`: `compute_drift()` | Uses `SequenceMatcher` for LCS ratio |
| Drift bound check (delta <= 0.18) | `orchestrator/app.py`: `within_drift_bound()` | Compare `POAAS_ABLATION=full` vs `POAAS_ABLATION=no_drift` |
| Length expansion bound (rho <= 2.4) | `orchestrator/app.py`: `within_length_bound()` | Enforced in `merge_outputs()` |
| Merge order: Cleaner -> Paraphraser -> Facts | `orchestrator/app.py`: `merge_outputs()` | Sequential application with drift checks |

## Experiments and Results

### Table 1: Main Results (Clean Inputs)

| What | Location | Command |
|------|----------|---------|
| Full experiment runner | `scripts/run_full_experiments.py` | `python scripts/run_full_experiments.py` |
| Quick reproduction | `scripts/run_experiments.py` | `python scripts/run_experiments.py --config manuscript_full` |
| Aggregate into tables | `scripts/aggregate_results.py` | `python scripts/aggregate_results.py` |
| Example output | `results/tables/table3.tex` | Pre-generated from test run |

### Table 2: Degradation Robustness

| What | Location | Command |
|------|----------|---------|
| Noise injection (deletion, mixup) | `eval/noise.py` | Imported by `scripts/run_full_experiments.py` |
| Run degraded experiments | `scripts/run_full_experiments.py` | Automatically tests 5%, 10%, 15% noise |

### Table 3: Efficiency Metrics

| What | Location | Command |
|------|----------|---------|
| Latency tracking | `orchestrator/app.py` | Measured per-request in `InferResponse.latency_ms` |
| Prometheus metrics | `poaas/common/metrics.py` | `LATENCY_HISTOGRAM`, `WORKER_CALLS_COUNTER` |
| Result aggregation | `scripts/aggregate_results.py` | `make aggregate` |

### Table 4: Ablation Study

| Ablation | Environment Variable | Command |
|----------|---------------------|---------|
| Full system | `POAAS_ABLATION=full` | `python scripts/run_experiments.py --config ablation` |
| Without skip logic | `POAAS_ABLATION=no_skip` | `make ablation-no-skip` |
| Without drift control | `POAAS_ABLATION=no_drift` | `make ablation-no-drift` |

### Table 5: Cross-Model Results

| Model | HF Identifier | Download |
|-------|---------------|----------|
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | `bash scripts/download_models.sh 3b` |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | `bash scripts/download_models.sh 8b` |

## Baselines

| Baseline | Paper Reference | Implementation |
|----------|----------------|----------------|
| EvoPrompt | Chen et al., 2023 | `baselines/evoprompt.py` |
| OPRO | Yang et al., 2023 | `baselines/opro.py` |
| PromptWizard | Agarwal et al., 2024 | `baselines/promptwizard.py` |
| APO | Pryzant et al., 2023 | `baselines/apo.py` |

## Benchmarks

| Benchmark | Type | Dataset Source | Download |
|-----------|------|---------------|----------|
| BBH | Reasoning | `lukaemon/bbh` (Hugging Face) | `python scripts/download_datasets.py` |
| GSM8K | Math reasoning | `openai/gsm8k` (Hugging Face) | `python scripts/download_datasets.py` |
| CommonsenseQA | Commonsense | `tau/commonsense_qa` (Hugging Face) | `python scripts/download_datasets.py` |
| HaluEval | Hallucination detection | `pminervini/HaluEval` (Hugging Face) | `python scripts/download_datasets.py` |
| HalluLens | Consistency | Template-generated | `python scripts/download_datasets.py` |
| FActScore | Factual accuracy | Template-generated | `python scripts/download_datasets.py` |

## Evaluation

| Component | Location | Description |
|-----------|----------|-------------|
| Real evaluation framework | `eval/real_evaluation.py` | End-to-end eval with actual model inference |
| Noise injection | `eval/noise.py` | Token deletion and mixup at configurable rates |
| LLM judge (HaluEval/HalluLens) | `eval/llm_judge.py` | GPT-based hallucination judging |
| Cost computation | `config/prices.yml` | Per-token pricing for cost analysis |

## Not Yet Implemented

| Component | What's Needed |
|-----------|---------------|
| Vectara HHEM integration | Integrate Vectara's Hallucination and Heritage Evaluation Model (HHEM) API for factuality scoring; add optional HHEM-based evaluation path alongside existing LLM judge. |
| 95% bootstrap confidence intervals | Add bootstrap resampling (e.g., via `scipy.stats.bootstrap` or manual resampling) to aggregate results; report mean ± 95% CI for each metric in tables. |
| PromptBreeder baseline | Implement PromptBreeder baseline (Fernando et al.) as a new optimizer in `baselines/`; add to `--methods` choices in evaluation harness. |
