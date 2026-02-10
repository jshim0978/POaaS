#!/usr/bin/env python3
"""
LLM-as-a-Judge evaluation for POaaS factuality benchmarks.

Implements the GPT-5-as-a-Judge protocol from Appendix H:
- Strict binary hallucination detection
- Temperature 0.0 for deterministic outputs
- Exact prompts from the paper

Usage:
    from eval.llm_judge import HallucinationJudge
    
    judge = HallucinationJudge(model="gpt-4o")  # or other OpenAI-compatible model
    result = await judge.evaluate(answer, context, gold_answer, task)
"""

import asyncio
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Try to import the vllm_client, fallback to direct API call
try:
    from poaas.common.vllm_client import chat
    HAS_VLLM_CLIENT = True
except ImportError:
    HAS_VLLM_CLIENT = False

import httpx


class JudgmentResult(Enum):
    """Possible judgment outcomes."""
    NOT_HALLUCINATED = "not_hallucinated"
    HALLUCINATED = "hallucinated"
    ERROR = "error"


@dataclass
class JudgeOutput:
    """Output from the hallucination judge."""
    judgment: JudgmentResult
    raw_response: str
    is_factual: bool  # Convenience: True if NOT_HALLUCINATED
    confidence: float  # 1.0 for clear judgment, 0.5 for unclear


# Judge prompts from Appendix H
JUDGE_SYSTEM_PROMPT = """You are a strict hallucination judge.
Given a model answer and optional reference context, output exactly one token:
hallucinated if the answer introduces unsupported or contradicted facts;
not_hallucinated if it is supported by the context/evidence.
Do not explain."""

JUDGE_USER_TEMPLATE = """Task: {task}
Evidence (if any):
{evidence}

Gold / reference answer (if provided):
{gold}

Model answer:
{answer}

Output exactly one token: 'hallucinated' or 'not_hallucinated'."""


class HallucinationJudge:
    """
    LLM-based hallucination judge implementing the paper's evaluation protocol.
    
    Supports:
    - OpenAI API (GPT-4o, GPT-3.5-turbo, etc.)
    - vLLM-compatible APIs
    - Local models via vllm_client
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 10
    ):
        """
        Initialize the hallucination judge.
        
        Args:
            model: Model to use for judgment (e.g., "gpt-4o", "gpt-3.5-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            api_base: Custom API base URL (for vLLM or other compatible APIs)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Max tokens to generate (small since we only need one word)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def _format_prompt(
        self,
        answer: str,
        task: str = "",
        evidence: str = "",
        gold: str = ""
    ) -> str:
        """Format the user prompt for the judge."""
        return JUDGE_USER_TEMPLATE.format(
            task=task or "General question answering",
            evidence=evidence or "(none provided)",
            gold=gold or "(not provided)",
            answer=answer
        )
    
    def _parse_judgment(self, response: str) -> JudgmentResult:
        """Parse the judge's response into a judgment."""
        response_lower = response.strip().lower()
        
        # Check for exact matches
        if "not_hallucinated" in response_lower or "not hallucinated" in response_lower:
            return JudgmentResult.NOT_HALLUCINATED
        elif "hallucinated" in response_lower:
            return JudgmentResult.HALLUCINATED
        
        # Fuzzy matching for common variations
        if any(word in response_lower for word in ["factual", "accurate", "correct", "supported", "true"]):
            return JudgmentResult.NOT_HALLUCINATED
        elif any(word in response_lower for word in ["false", "incorrect", "unsupported", "fabricated"]):
            return JudgmentResult.HALLUCINATED
        
        # Default to error if unclear
        return JudgmentResult.ERROR
    
    async def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI-compatible API."""
        if not self.api_key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY or pass api_key to constructor.")
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _call_vllm_client(self, messages: List[Dict[str, str]]) -> str:
        """Call via vllm_client if available."""
        if not HAS_VLLM_CLIENT:
            raise ImportError("vllm_client not available")
        
        result = await chat(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model
        )
        return result["text"]
    
    async def evaluate(
        self,
        answer: str,
        task: str = "",
        evidence: str = "",
        gold: str = ""
    ) -> JudgeOutput:
        """
        Evaluate whether an answer contains hallucinations.
        
        Args:
            answer: The model's answer to evaluate
            task: Description of the task (optional)
            evidence: Context/evidence to check against (optional)
            gold: Gold/reference answer (optional)
            
        Returns:
            JudgeOutput with judgment, raw response, and convenience fields
        """
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self._format_prompt(answer, task, evidence, gold)}
        ]
        
        try:
            # Try OpenAI API first
            if self.api_key:
                raw_response = await self._call_openai_api(messages)
            elif HAS_VLLM_CLIENT:
                raw_response = await self._call_vllm_client(messages)
            else:
                # Fallback to heuristic judgment
                raw_response = self._heuristic_judgment(answer, evidence, gold)
            
            judgment = self._parse_judgment(raw_response)
            
            return JudgeOutput(
                judgment=judgment,
                raw_response=raw_response,
                is_factual=(judgment == JudgmentResult.NOT_HALLUCINATED),
                confidence=1.0 if judgment != JudgmentResult.ERROR else 0.5
            )
            
        except Exception as e:
            return JudgeOutput(
                judgment=JudgmentResult.ERROR,
                raw_response=f"Error: {str(e)}",
                is_factual=False,
                confidence=0.0
            )
    
    def _heuristic_judgment(self, answer: str, evidence: str, gold: str) -> str:
        """
        Fallback heuristic judgment when no API is available.
        
        This is a simplified check - real evaluation should use LLM judge.
        """
        answer_lower = answer.lower()
        
        # Check if answer matches gold
        if gold and gold.lower() in answer_lower:
            return "not_hallucinated"
        
        # Check if answer is supported by evidence
        if evidence:
            evidence_lower = evidence.lower()
            # Simple keyword overlap check
            answer_words = set(answer_lower.split())
            evidence_words = set(evidence_lower.split())
            overlap = len(answer_words & evidence_words) / max(len(answer_words), 1)
            
            if overlap > 0.3:
                return "not_hallucinated"
        
        # Check for hedging language (less likely to be hallucination)
        hedging = ["i don't know", "i'm not sure", "uncertain", "may be", "might be", "possibly"]
        if any(h in answer_lower for h in hedging):
            return "not_hallucinated"
        
        # Default to hallucinated if no evidence of factuality
        return "hallucinated"
    
    async def evaluate_batch(
        self,
        items: List[Dict[str, str]],
        concurrent_limit: int = 5
    ) -> List[JudgeOutput]:
        """
        Evaluate a batch of items concurrently.
        
        Args:
            items: List of dicts with 'answer' and optional 'task', 'evidence', 'gold'
            concurrent_limit: Max concurrent API calls
            
        Returns:
            List of JudgeOutput in same order as input
        """
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def evaluate_with_limit(item: Dict[str, str]) -> JudgeOutput:
            async with semaphore:
                return await self.evaluate(
                    answer=item.get("answer", ""),
                    task=item.get("task", ""),
                    evidence=item.get("evidence", ""),
                    gold=item.get("gold", "")
                )
        
        tasks = [evaluate_with_limit(item) for item in items]
        return await asyncio.gather(*tasks)


# Benchmark-specific evaluation functions

async def evaluate_halueval(
    judge: HallucinationJudge,
    response: str,
    sample: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a response for HaluEval benchmark.
    
    HaluEval provides knowledge context and expects factual answers.
    """
    result = await judge.evaluate(
        answer=response,
        task="HaluEval QA - Answer based on provided knowledge",
        evidence=sample.get("knowledge", ""),
        gold=sample.get("answer", "")
    )
    
    return {
        "is_factual": result.is_factual,
        "judgment": result.judgment.value,
        "confidence": result.confidence
    }


async def evaluate_hallulens(
    judge: HallucinationJudge,
    response: str,
    sample: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a response for HalluLens benchmark.
    
    HalluLens tests consistency and refusal on non-existent entities.
    """
    category = sample.get("category", "")
    
    if category == "NonExistentRefusal":
        # For non-existent entities, should refuse to answer
        refusal_phrases = [
            "don't have information", "don't know", "cannot find",
            "no information", "not aware", "unable to", "doesn't exist"
        ]
        is_correct_refusal = any(phrase in response.lower() for phrase in refusal_phrases)
        
        return {
            "is_factual": is_correct_refusal,
            "judgment": "not_hallucinated" if is_correct_refusal else "hallucinated",
            "confidence": 1.0
        }
    else:
        # For wiki QA, check factuality
        result = await judge.evaluate(
            answer=response,
            task=f"HalluLens {category} - Verify factual accuracy",
            evidence=sample.get("context", ""),
            gold=sample.get("answer", "")
        )
        
        return {
            "is_factual": result.is_factual,
            "judgment": result.judgment.value,
            "confidence": result.confidence
        }


async def evaluate_factscore(
    judge: HallucinationJudge,
    response: str,
    sample: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a response for FActScore benchmark.
    
    FActScore evaluates factual accuracy of generated biographies.
    Note: Full FActScore requires atomic fact decomposition; this is simplified.
    """
    entity = sample.get("entity", "")
    
    result = await judge.evaluate(
        answer=response,
        task=f"FActScore - Biography of {entity}",
        evidence="",  # FActScore typically uses retrieved evidence
        gold=""
    )
    
    return {
        "is_factual": result.is_factual,
        "judgment": result.judgment.value,
        "confidence": result.confidence,
        "entity": entity
    }


def compute_factuality_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute aggregate factuality metrics from individual results.
    
    Returns:
        Dict with 'accuracy', 'factual_rate', 'avg_confidence'
    """
    if not results:
        return {"accuracy": 0.0, "factual_rate": 0.0, "avg_confidence": 0.0}
    
    factual_count = sum(1 for r in results if r.get("is_factual", False))
    confidences = [r.get("confidence", 0.0) for r in results]
    
    return {
        "accuracy": factual_count / len(results),
        "factual_rate": factual_count / len(results),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "total_samples": len(results),
        "factual_samples": factual_count
    }


if __name__ == "__main__":
    # Test the judge
    async def test():
        judge = HallucinationJudge(model="gpt-3.5-turbo")
        
        # Test case 1: Factual answer
        result1 = await judge.evaluate(
            answer="Paris is the capital of France.",
            task="Geography question",
            evidence="France is a country in Western Europe. Its capital is Paris.",
            gold="Paris"
        )
        print(f"Test 1 (factual): {result1.judgment.value} (expected: not_hallucinated)")
        
        # Test case 2: Hallucinated answer
        result2 = await judge.evaluate(
            answer="The capital of France is Berlin.",
            task="Geography question",
            evidence="France is a country in Western Europe. Its capital is Paris.",
            gold="Paris"
        )
        print(f"Test 2 (hallucination): {result2.judgment.value} (expected: hallucinated)")
        
        # Test case 3: Refusal (for non-existent entity)
        result3 = await judge.evaluate(
            answer="I don't have information about that entity.",
            task="Biography request for fictional entity",
            evidence="",
            gold=""
        )
        print(f"Test 3 (refusal): {result3.judgment.value} (expected: not_hallucinated)")
    
    asyncio.run(test())

