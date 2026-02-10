"""
POaaS Common Utilities

Provides shared functionality across POaaS services:
- vllm_client: LLM inference client (vLLM/OpenAI compatible)
- config: Centralized configuration loading
- artifacts: Run artifact persistence
- metrics: Prometheus metrics
"""

from poaas.common.vllm_client import chat, chat_sync, chat_with_worker_config

try:
    from poaas.common.config import (
        get_config,
        get_model_id,
        get_default_model,
        get_decoding_params,
        get_poaas_config,
        get_ablation_config,
        get_worker_config,
        get_seed,
        get_temperature,
        get_max_tokens,
        get_top_p,
        get_skip_threshold,
        get_max_drift,
        get_max_length_ratio,
        get_fact_token_limit,
        get_max_facts
    )
except ImportError:
    pass

try:
    from poaas.common.artifacts import (
        generate_run_id,
        get_run_path,
        save_run_artifact,
        save_input_artifact,
        save_output_artifact,
        save_final_artifact,
        load_run_artifact,
        list_runs,
        get_run_summary
    )
except ImportError:
    pass

try:
    from poaas.common.metrics import (
        record_request,
        record_worker_call,
        record_drift,
        record_quality,
        get_metrics_response,
        get_metrics_app,
        timed_request
    )
except ImportError:
    pass

__all__ = [
    "chat",
    "chat_sync",
    "chat_with_worker_config",
    "get_config",
    "get_model_id",
    "get_default_model",
    "get_decoding_params",
    "get_poaas_config",
    "generate_run_id",
    "save_run_artifact",
    "record_request",
    "get_metrics_app"
]
