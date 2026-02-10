"""
Paraphraser Worker for POaaS.

Improves clarity and fluency of user prompts.
Uses configuration from centralized config module.
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI
from pydantic import BaseModel

from poaas.common.vllm_client import chat_with_worker_config

app = FastAPI(title="Paraphraser Worker")

# System prompt matching paper Appendix E.2
PARA_SYS = (
    "You are a paraphrasing agent. Rephrase the input for clarity and naturalness. "
    "Preserve the original meaning and intent. Make it more fluent and readable. "
    "Output only the paraphrased text."
)


class ParaIn(BaseModel):
    text: str


class ParaOut(BaseModel):
    paraphrased: str
    latency_ms: float


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "service": "paraphraser"}


@app.post("/paraphrase", response_model=ParaOut)
async def paraphrase(inp: ParaIn):
    """
    Paraphrase a prompt for improved clarity.
    
    Uses temperature=0.3 for slight variation.
    """
    msgs = [
        {"role": "system", "content": PARA_SYS},
        {"role": "user", "content": inp.text},
    ]
    
    out = await chat_with_worker_config(msgs, "paraphraser")
    
    return ParaOut(
        paraphrased=out["text"],
        latency_ms=out["latency_ms"]
    )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Paraphraser Worker")
    parser.add_argument("--port", type=int, default=8003, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
