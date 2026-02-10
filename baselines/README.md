# POaaS Baseline Methods

This directory contains implementations of state-of-the-art prompt optimization baseline methods for comparison with POaaS.

## Implemented Baselines

### EvoPrompt
- **Paper**: "Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers" (EMNLP 2023)
- **Method**: Population-based evolutionary algorithm with LLM-guided mutations and crossover
- **Key Features**:
  - Population size: 6-8 candidates per generation
  - Generations: 8-10 evolution rounds
  - Differential evolution operators
  - Fitness evaluation on task examples

### OPRO (Optimization by Prompting)  
- **Paper**: "Large Language Models as Optimizers" (ICML 2024)
- **Method**: Meta-prompting approach using LLM as optimizer
- **Key Features**:
  - Meta-optimization with performance history
  - Iterative prompt improvement (6-8 iterations)
  - 4-5 candidates per iteration
  - LLM-guided optimization trajectory

### PromptWizard
- **Paper**: "PromptWizard: Task-Aware Agent-driven Prompt Optimization Framework" (EMNLP 2024)
- **Method**: Task-aware critique and synthesis approach
- **Key Features**:
  - Multi-agent optimization (critic + synthesizer)
  - 3-4 optimization rounds
  - Task-specific analysis and improvement
  - Iterative critique-synthesis cycles

### APO (Automatic Prompt Optimization)
- **Paper**: "Automatic Prompt Optimization with 'Gradient Descent' and Beam Search" (EMNLP 2023)
- **Method**: Textual gradients with beam search optimization
- **Key Features**:
  - Textual gradient computation from errors
  - Beam search over prompt candidates
  - Minibatch-based optimization (8 samples)
  - LLM-guided prompt editing

## Usage

### Individual Testing
```bash
# Test individual baselines
python baselines/test_baseline.py evoprompt
python baselines/test_baseline.py opro  
python baselines/test_baseline.py promptwizard
python baselines/test_baseline.py apo
```

### Full Evaluation
```bash
# Compare all baselines on multiple benchmarks
python scripts/run_experiments.py --config test --limit 10

# Run manuscript replication with all methods
python scripts/run_experiments.py --config manuscript_full
```

## Implementation Details

All baselines are implemented with:
- **Functional optimization algorithms** that call the vLLM API
- **Async/sync compatibility** for the evaluation framework
- **Error handling** with fallback mechanisms
- **Configurable parameters** for different evaluation scenarios
- **Integration** with the POaaS evaluation pipeline

## Performance Characteristics

| Method | Optimization Time | Complexity | Approach |
|--------|------------------|-----------|----------|
| POaaS | ~125ms | Low | CPU-only heuristics |
| EvoPrompt | Variable | High | Population evolution |
| OPRO | Variable | Medium | Meta-prompting |
| PromptWizard | Variable | High | Multi-agent |
| APO | Variable | Medium | Textual gradients |

## References

- EvoPrompt: [https://arxiv.org/abs/2309.08532](https://arxiv.org/abs/2309.08532)
- OPRO: [https://arxiv.org/abs/2309.03409](https://arxiv.org/abs/2309.03409)  
- PromptWizard: [https://arxiv.org/abs/2405.12877](https://arxiv.org/abs/2405.12877)
- APO: [https://aclanthology.org/2023.emnlp-main.494/](https://aclanthology.org/2023.emnlp-main.494/)