"""
Cleaner Worker for POaaS.

Fixes typos, grammar, and spacing issues in user prompts.
Uses configuration from centralized config module.
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI
from pydantic import BaseModel

from poaas.common.vllm_client import chat_with_worker_config

app = FastAPI(title="Cleaner Worker")

# System prompt matching paper Appendix E.1
CLEAN_SYS = (
    "You are a prompt-cleaning agent. Fix typos, casing, spacing, and grammar. "
    "Preserve meaning. Do NOT add new facts. Output only the corrected text."
)


class CleanIn(BaseModel):
    text: str


class CleanOut(BaseModel):
    cleaned: str
    latency_ms: float


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "service": "cleaner"}


@app.post("/clean", response_model=CleanOut)
async def clean(inp: CleanIn):
    """
    Clean a prompt by fixing typos and grammar.
    
    Uses temperature=0.0 for deterministic output.
    """
    msgs = [
        {"role": "system", "content": CLEAN_SYS},
        {"role": "user", "content": inp.text},
    ]
    
    out = await chat_with_worker_config(msgs, "cleaner")
    
    return CleanOut(
        cleaned=out["text"],
        latency_ms=out["latency_ms"]
    )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Cleaner Worker")
    parser.add_argument("--port", type=int, default=8002, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
