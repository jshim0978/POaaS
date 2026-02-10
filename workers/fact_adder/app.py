"""
Fact-Adder Worker for POaaS.

Adds concise factual context to user prompts.
Uses configuration from centralized config module.
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import FastAPI
from pydantic import BaseModel

from poaas.common.vllm_client import chat_with_worker_config

app = FastAPI(title="Fact-Adder Worker")

# System prompt matching paper Appendix E.3
FACT_SYS = (
    "You are a fact-generation agent. Given a user query, generate up to 3 concise factual "
    "bullets that provide relevant context. Each fact must be:\n"
    "- Directly related to entities/topics in the query\n"
    "- Factually accurate and verifiable\n"
    "- Concise (approximately one sentence each)\n"
    "- Formatted as bullet points\n\n"
    "Output ONLY the factual bullets, nothing else. If no relevant facts can be confidently "
    "provided, output 'NONE'. Do NOT answer the question directly."
)


class FactIn(BaseModel):
    text: str


class FactOut(BaseModel):
    facts: str
    latency_ms: float


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "service": "fact_adder"}


@app.post("/facts", response_model=FactOut)
async def add_facts(inp: FactIn):
    """
    Generate factual context for a prompt.
    
    Uses temperature=0.0 for deterministic fact generation.
    Limited to 120 tokens per config.
    """
    msgs = [
        {"role": "system", "content": FACT_SYS},
        {"role": "user", "content": inp.text},
    ]
    
    out = await chat_with_worker_config(msgs, "fact_adder")
    facts = out["text"].strip()
    
    # If model outputs "NONE" or empty, return empty facts
    if facts.upper() == "NONE" or not facts:
        facts = ""
    
    return FactOut(
        facts=facts,
        latency_ms=out["latency_ms"]
    )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Fact-Adder Worker")
    parser.add_argument("--port", type=int, default=8004, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
