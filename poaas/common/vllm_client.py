"""
vLLM client for POaaS workers.

Supports both OpenAI-compatible APIs and local vLLM servers.
Reads configuration from centralized config module.

Usage:
    from poaas.common.vllm_client import chat
    
    result = await chat([
        {"role": "user", "content": "Hello!"}
    ])
    print(result["text"])
"""

import asyncio
import httpx
import os
import time
from typing import Dict, List, Optional, Any, Tuple

# Try to import config, fallback to defaults if not available
try:
    from poaas.common.config import (
        get_default_model, get_decoding_params, get_worker_config
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


def _get_default_params() -> Dict[str, Any]:
    """Get default parameters from config or fallback."""
    if HAS_CONFIG:
        params = get_decoding_params()
        return {
            "model": get_default_model(),
            "temperature": params.get("temperature", 0.2),
            "max_tokens": params.get("max_tokens", 512),
            "top_p": params.get("top_p", 0.9),
            "seed": params.get("seed", 13)
        }
    else:
        return {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 0.9,
            "seed": 13
        }


async def chat(
    messages: List[Dict[str, str]], 
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Send chat completion request to vLLM server or OpenAI API.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature (uses config default if None)
        max_tokens: Maximum tokens to generate (uses config default if None)
        top_p: Top-p sampling (uses config default if None)
        model: Model name (uses config default if None)
        seed: Random seed (uses config default if None)
        
    Returns:
        Dict with 'text', 'latency_ms', and 'usage' keys
    """
    start_time = time.perf_counter()
    
    # Get defaults from config
    defaults = _get_default_params()
    
    # Apply defaults for None values
    if model is None:
        model = defaults["model"]
    if temperature is None:
        temperature = defaults["temperature"]
    if max_tokens is None:
        max_tokens = defaults["max_tokens"]
    if top_p is None:
        top_p = defaults["top_p"]
    if seed is None:
        seed = defaults["seed"]
    
    # Check for API endpoint configuration
    vllm_url = os.getenv("VLLM_URL", "http://localhost:8000")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        if openai_api_key:
            # Use OpenAI API
            response_text, usage = await _call_openai_api(
                messages, temperature, max_tokens, top_p, openai_api_key
            )
        else:
            # Use local vLLM server
            response_text, usage = await _call_vllm_server(
                messages, temperature, max_tokens, top_p, model, vllm_url, seed
            )
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "text": response_text.strip(),
            "latency_ms": latency_ms,
            "usage": usage
        }
        
    except Exception as e:
        # Fallback to simple echo for testing
        latency_ms = (time.perf_counter() - start_time) * 1000
        user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        
        return {
            "text": f"[FALLBACK] {user_content}",
            "latency_ms": latency_ms,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": str(e)
        }


async def _call_openai_api(
    messages: List[Dict[str, str]], 
    temperature: float, 
    max_tokens: int,
    top_p: float,
    api_key: str
) -> Tuple[str, Dict[str, int]]:
    """Call OpenAI ChatCompletion API."""
    # Use GPT model when using OpenAI API
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{api_base}/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        result = response.json()
        usage = result.get("usage", {})
        
        return result["choices"][0]["message"]["content"], usage


async def _call_vllm_server(
    messages: List[Dict[str, str]], 
    temperature: float, 
    max_tokens: int,
    top_p: float,
    model: str,
    base_url: str,
    seed: int
) -> Tuple[str, Dict[str, int]]:
    """Call local vLLM server."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "seed": seed,
        "stream": False
    }
    
    timeout_sec = float(os.getenv("VLLM_TIMEOUT", "60"))
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        usage = result.get("usage", {})
        
        return result["choices"][0]["message"]["content"], usage


async def chat_with_worker_config(
    messages: List[Dict[str, str]],
    worker_name: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send chat request using worker-specific configuration.
    
    Args:
        messages: List of message dicts
        worker_name: One of "cleaner", "paraphraser", "fact_adder"
        model: Override model (uses default if None)
        
    Returns:
        Dict with 'text', 'latency_ms', and 'usage'
    """
    if HAS_CONFIG:
        worker_config = get_worker_config(worker_name)
        temperature = worker_config.get("temperature")
        max_tokens = worker_config.get("max_tokens")
    else:
        # Fallback worker configs
        worker_defaults = {
            "cleaner": {"temperature": 0.0, "max_tokens": 512},
            "paraphraser": {"temperature": 0.3, "max_tokens": 512},
            "fact_adder": {"temperature": 0.0, "max_tokens": 120}
        }
        config = worker_defaults.get(worker_name, {"temperature": 0.2, "max_tokens": 512})
        temperature = config["temperature"]
        max_tokens = config["max_tokens"]
    
    return await chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model
    )


# Synchronous wrapper for non-async contexts
def chat_sync(
    messages: List[Dict[str, str]], 
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Synchronous version of chat() for non-async contexts."""
    return asyncio.run(chat(messages, temperature, max_tokens, model=model))


if __name__ == "__main__":
    # Test the client
    async def test():
        result = await chat([
            {"role": "user", "content": "What is 2 + 2?"}
        ])
        print(f"Response: {result['text']}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
    
    asyncio.run(test())
