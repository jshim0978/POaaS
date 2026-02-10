# Migration Notes: MPR-SaaS to POaaS

This document records the transformation from the internal MPR-SaaS codebase to
the public POaaS repository.

## Lineage

- **MPR-SaaS** (internal): 2-agent system with Cleaner (typo + keyword) and
  Describer (JSON spec + summary). Deployed on three internal GPU nodes behind
  HAProxy load balancing with JWT authentication.

- **POaaS** (public): 3-agent system with Cleaner, Paraphraser, and Fact-Adder.
  Extended with drift-controlled merging, conservative skip logic, CPU-only
  heuristic routing, and four baseline implementations.

## Key Changes

### Architecture

| Aspect | MPR-SaaS (internal) | POaaS (public) |
|--------|---------------------|----------------|
| Workers | Cleaner + Describer | Cleaner + Paraphraser + Fact-Adder |
| Routing | Simple ill-formedness score | 4-dimension quality scoring (typo, completeness, fluency, clarity) |
| Skip logic | Threshold on ill-formedness | Conservative skip with quality > 0.75 and typo < 0.20 |
| Merging | Concatenation | Drift-controlled with delta <= 0.18 and rho <= 2.4 |
| Deployment | HAProxy + JWT on internal nodes | Docker Compose or local services |
| Baselines | None | EvoPrompt, OPRO, PromptWizard, APO |
| Evaluation | None | 6 benchmarks with noise injection |

### Removed from Public Release

- Internal IP addresses and hostnames
- HAProxy configuration and JWT authentication details
- Cursor IDE rules (`.cursor/rules/`)
- Deployment scripts referencing internal infrastructure
- LoRA adapter references (training-only artifacts)
- Paper PDF (link to proceedings instead)
- Mock evaluation file with hardcoded manuscript results
- Stale experiment result files from test runs

### Naming

The manuscript uses "PRaaS" (Prompt Refinement as a Service) in some LaTeX
source files. The public repository and all user-facing documentation use
"POaaS" (Prompt Optimization as a Service). Both refer to the same system.
