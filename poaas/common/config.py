"""
Centralized configuration loading for POaaS.

Loads configuration from config/decoding.json and provides
consistent access to model, decoding, and POaaS parameters.

Usage:
    from poaas.common.config import get_config, get_model_id, get_decoding_params
    
    config = get_config()
    model = get_model_id("3b")  # or "8b"
    params = get_decoding_params()
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


# Find config file relative to package or workspace root
def _find_config_path() -> Path:
    """Find the config file path."""
    # Try relative to this file
    pkg_root = Path(__file__).parent.parent.parent
    config_path = pkg_root / "config" / "decoding.json"
    
    if config_path.exists():
        return config_path
    
    # Try from current working directory
    cwd_config = Path("config/decoding.json")
    if cwd_config.exists():
        return cwd_config
    
    # Try from environment variable
    env_config = os.getenv("POAAS_CONFIG")
    if env_config:
        env_path = Path(env_config)
        if env_path.exists():
            return env_path
    
    raise FileNotFoundError(
        "Could not find config/decoding.json. "
        "Set POAAS_CONFIG environment variable or run from project root."
    )


@lru_cache(maxsize=1)
def get_config() -> Dict[str, Any]:
    """
    Load and cache the full configuration.
    
    Returns:
        Dict containing all configuration values
    """
    config_path = _find_config_path()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def reload_config() -> Dict[str, Any]:
    """Force reload of configuration (clears cache)."""
    get_config.cache_clear()
    return get_config()


def get_model_id(model_size: str = "3b") -> str:
    """
    Get the model ID for a given size.
    
    Args:
        model_size: One of "3b" or "8b"
        
    Returns:
        Full model ID string (e.g., "meta-llama/Llama-3.2-3B-Instruct")
    """
    config = get_config()
    models = config.get("models", {})
    
    if model_size in models:
        return models[model_size]
    
    # Fallback to default
    default_size = config.get("default_model", "3b")
    return models.get(default_size, "meta-llama/Llama-3.2-3B-Instruct")


def get_default_model() -> str:
    """Get the default model ID."""
    config = get_config()
    default_size = config.get("default_model", "3b")
    return get_model_id(default_size)


def get_decoding_params() -> Dict[str, Any]:
    """
    Get decoding parameters.
    
    Returns:
        Dict with temperature, top_p, max_tokens, seed
    """
    config = get_config()
    return config.get("decoding", {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 512,
        "seed": 13
    })


def get_poaas_config() -> Dict[str, Any]:
    """
    Get POaaS-specific configuration.
    
    Returns:
        Dict with thresholds, drift bounds, fact limits
    """
    config = get_config()
    return config.get("poaas_config", {
        "typo_threshold": 0.30,
        "completeness_threshold": 0.70,
        "fluency_threshold": 0.80,
        "clarity_threshold": 0.70,
        "skip_threshold": 0.25,
        "max_drift": 0.18,
        "max_length_ratio": 2.4,
        "fact_token_limit": 120,
        "max_facts": 3
    })


def get_ablation_config(ablation_name: str = "full") -> Dict[str, Any]:
    """
    Get ablation configuration.
    
    Args:
        ablation_name: One of "full", "no_skip", "no_drift"
        
    Returns:
        Dict with skip_enabled and drift_enabled flags
    """
    config = get_config()
    ablations = config.get("ablations", {})
    
    return ablations.get(ablation_name, {
        "skip_enabled": True,
        "drift_enabled": True,
        "description": "Default full configuration"
    })


def get_worker_config(worker_name: str) -> Dict[str, Any]:
    """
    Get worker-specific configuration.
    
    Args:
        worker_name: One of "cleaner", "paraphraser", "fact_adder"
        
    Returns:
        Dict with temperature, max_tokens for the worker
    """
    config = get_config()
    worker_configs = config.get("worker_config", {})
    
    # Default worker configs
    defaults = {
        "cleaner": {"temperature": 0.0, "max_tokens": 512},
        "paraphraser": {"temperature": 0.3, "max_tokens": 512},
        "fact_adder": {"temperature": 0.0, "max_tokens": 120}
    }
    
    return worker_configs.get(worker_name, defaults.get(worker_name, {}))


# Convenience accessors for common values
def get_seed() -> int:
    """Get the random seed."""
    return get_decoding_params().get("seed", 13)


def get_temperature() -> float:
    """Get the default temperature."""
    return get_decoding_params().get("temperature", 0.2)


def get_max_tokens() -> int:
    """Get the default max tokens."""
    return get_decoding_params().get("max_tokens", 512)


def get_top_p() -> float:
    """Get the default top_p."""
    return get_decoding_params().get("top_p", 0.9)


# POaaS-specific accessors
def get_skip_threshold() -> float:
    """Get skip logic threshold."""
    return get_poaas_config().get("skip_threshold", 0.25)


def get_max_drift() -> float:
    """Get maximum drift threshold."""
    return get_poaas_config().get("max_drift", 0.18)


def get_max_length_ratio() -> float:
    """Get maximum length expansion ratio."""
    return get_poaas_config().get("max_length_ratio", 2.4)


def get_fact_token_limit() -> int:
    """Get fact token limit."""
    return get_poaas_config().get("fact_token_limit", 120)


def get_max_facts() -> int:
    """Get maximum number of facts."""
    return get_poaas_config().get("max_facts", 3)


if __name__ == "__main__":
    # Test config loading
    print("Configuration Test")
    print("=" * 50)
    
    config = get_config()
    print(f"Config loaded successfully")
    print(f"Models: {config.get('models', {})}")
    print(f"Default model: {get_default_model()}")
    print(f"Decoding params: {get_decoding_params()}")
    print(f"POaaS config: {get_poaas_config()}")
    print(f"Ablation 'no_skip': {get_ablation_config('no_skip')}")
    print(f"Worker config 'cleaner': {get_worker_config('cleaner')}")

