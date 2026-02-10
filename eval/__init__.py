"""
POaaS Evaluation Framework

Provides:
- noise: Input degradation (token deletion, mixup)
- llm_judge: LLM-as-judge for factuality evaluation
- real_evaluation: Full evaluation framework
"""

try:
    from eval.noise import apply_noise, NoiseConfig, get_noise_conditions
except ImportError:
    pass

try:
    from eval.llm_judge import (
        HallucinationJudge,
        evaluate_halueval,
        evaluate_hallulens,
        evaluate_factscore
    )
except ImportError:
    pass

__all__ = [
    "apply_noise",
    "NoiseConfig",
    "get_noise_conditions",
    "HallucinationJudge",
    "evaluate_halueval",
    "evaluate_hallulens",
    "evaluate_factscore"
]

