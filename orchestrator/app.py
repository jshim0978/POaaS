"""
POaaS Orchestrator - Minimal-Edit Prompt Optimization as a Service

Implements the POaaS system described in the FEVER 2026 paper:
1. CPU-only heuristic scoring (typo, completeness, fluency, clarity)
2. Conservative skip logic for high-quality prompts
3. Threshold-based routing to specialists
4. Drift-controlled merging with safety guards

Supports ablation modes:
- full: All components enabled (default)
- no_skip: Disable skip logic
- no_drift: Disable drift control
"""

import asyncio
import httpx
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from difflib import SequenceMatcher

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load .env file if present (allows 'cp .env.example .env' workflow)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

# Try to import config, fallback to defaults
try:
    from poaas.common.config import (
        get_poaas_config, get_ablation_config, get_max_facts, get_fact_token_limit
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False


def _get_poaas_params() -> Dict:
    """Get POaaS parameters from config or defaults."""
    if HAS_CONFIG:
        return get_poaas_config()
    return {
        "typo_threshold": 0.30,
        "completeness_threshold": 0.70,
        "fluency_threshold": 0.80,
        "clarity_threshold": 0.70,
        "skip_threshold": 0.25,
        "max_drift": 0.18,
        "max_length_ratio": 2.4,
        "fact_token_limit": 120,
        "max_facts": 3
    }


# Load configuration
_config = _get_poaas_params()

# POaaS Configuration (matching manuscript Table 5)
TYPO_THRESHOLD = _config.get("typo_threshold", 0.30)
COMPLETENESS_THRESHOLD = _config.get("completeness_threshold", 0.70)
FLUENCY_THRESHOLD = _config.get("fluency_threshold", 0.80)
CLARITY_THRESHOLD = _config.get("clarity_threshold", 0.70)
SKIP_THRESHOLD = _config.get("skip_threshold", 0.25)

# Drift and length constraints (matching manuscript Section 3.1)
MAX_DRIFT = _config.get("max_drift", 0.18)
MAX_LENGTH_RATIO = _config.get("max_length_ratio", 2.4)
FACT_TOKEN_LIMIT = _config.get("fact_token_limit", 120)
MAX_FACTS = _config.get("max_facts", 3)

# Worker endpoints
CLEANER_URL = os.getenv("CLEANER_URL", "http://localhost:8002")
PARAPHRASER_URL = os.getenv("PARAPHRASER_URL", "http://localhost:8003")
FACT_ADDER_URL = os.getenv("FACT_ADDER_URL", "http://localhost:8004")

# Ablation mode (can be set via environment or CLI)
ABLATION_MODE = os.getenv("POAAS_ABLATION", "full")

app = FastAPI(title="POaaS Orchestrator")


class InferRequest(BaseModel):
    prompt: str
    ablation: Optional[str] = None  # Override ablation mode per-request


class InferResponse(BaseModel):
    final_prompt: str
    skipped: bool
    workers_used: List[str]
    latency_ms: float
    reasoning: str
    ablation_mode: str = "full"
    run_id: Optional[str] = None


def get_ablation_flags(ablation_mode: str) -> Tuple[bool, bool]:
    """Get skip_enabled and drift_enabled flags for ablation mode."""
    if HAS_CONFIG:
        config = get_ablation_config(ablation_mode)
        return config.get("skip_enabled", True), config.get("drift_enabled", True)
    
    # Fallback ablation configs
    ablations = {
        "full": (True, True),
        "no_skip": (False, True),
        "no_drift": (True, False)
    }
    return ablations.get(ablation_mode, (True, True))


def compute_typo_score(text: str) -> float:
    """
    Compute typo score based on misspellings and character noise.
    
    Higher score = more typos detected.
    Paper reference: Section 3.2, Eq. 2
    """
    words = text.split()
    if not words:
        return 0.0
    
    # Common English words (lowercase) - expanded set for typo detection
    COMMON_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for',
        'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his',
        'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my',
        'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
        'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like',
        'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
        'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look',
        'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two',
        'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
        'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were', 'been',
        'has', 'had', 'did', 'does', 'where', 'why', 'many', 'much', 'more', 'each',
        'every', 'own', 'part', 'place', 'made', 'find', 'long', 'down', 'called',
        'capital', 'france', 'answer', 'question', 'please', 'provide', 'explain',
        'following', 'sentence', 'options', 'correct', 'true', 'false', 'none',
    }
    
    # Check each word for potential typos
    typo_indicators = 0
    word_count = 0
    
    for word in words:
        # Clean word (remove punctuation for checking)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if not clean_word or len(clean_word) < 2:
            continue
        
        word_count += 1
        
        # Skip if it's a common word
        if clean_word in COMMON_WORDS:
            continue
        
        # Check for typo patterns
        
        # 1. Too many consecutive consonants (unusual in English)
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{4,}', clean_word):
            typo_indicators += 1
            continue
        
        # 2. No vowels in long words (likely typo)
        if len(clean_word) > 3 and not re.search(r'[aeiouy]', clean_word):
            typo_indicators += 1
            continue
        
        # 3. Doubled consonants at word start (unusual)
        if re.match(r'^([bcdfghjklmnpqrstvwxz])\1', clean_word):
            typo_indicators += 0.5
            continue
        
        # 4. Check for common typo patterns (transpositions, missing letters)
        common_typos = {
            # Common transpositions
            'teh': 'the', 'waht': 'what', 'taht': 'that', 'thier': 'their',
            'th': 'the', 'wht': 'what', 'wher': 'where', 'wich': 'which',
            'hte': 'the', 'adn': 'and', 'nad': 'and', 'tow': 'two', 'owt': 'two',
            'hwo': 'how', 'hwat': 'what', 'whta': 'what', 'tihs': 'this',
            # Missing letters
            'becuase': 'because', 'recieve': 'receive', 'occured': 'occurred',
            'untill': 'until', 'definately': 'definitely', 'seperate': 'separate',
            'captial': 'capital', 'frace': 'france', 'anwser': 'answer',
            # Common misspellings
            'deos': 'does', 'dose': 'does', 'doesnot': 'does not',
            'thats': "that's", 'its': "it's", 'dont': "don't", 'wont': "won't",
            'acn': 'can', 'cna': 'can', 'aer': 'are', 'si': 'is', 'ti': 'it',
            'fo': 'of', 'ot': 'to', 'ont': 'not', 'nto': 'not',
            'wirte': 'write', 'wriet': 'write', 'tpye': 'type', 'tyep': 'type',
            'whit': 'with', 'wiht': 'with', 'wtih': 'with', 'iwth': 'with',
            'abotu': 'about', 'aobut': 'about', 'baout': 'about',
            'jsut': 'just', 'jstu': 'just', 'liek': 'like', 'lkie': 'like',
            'knwo': 'know', 'konw': 'know', 'kwon': 'know',
        }
        if clean_word in common_typos:
            typo_indicators += 1
            continue
        
        # 5. Very short words that aren't common (likely deletion artifacts)
        if len(clean_word) == 2 and clean_word not in COMMON_WORDS:
            typo_indicators += 0.3
    
    # Compute typo ratio
    if word_count == 0:
        return 0.0
    
    typo_ratio = typo_indicators / word_count
    
    # Check for missing punctuation in questions
    has_question_words = any(
        text.lower().startswith(qw) 
        for qw in ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'whom']
    )
    missing_punct = has_question_words and not text.strip().endswith('?')
    
    # Check for repeated characters (common typo pattern)
    repeated_chars = len(re.findall(r'(.)\1{2,}', text.lower()))
    repeated_ratio = repeated_chars / max(len(text), 1)
    
    # Combine indicators
    score = typo_ratio * 0.7 + repeated_ratio * 0.2
    if missing_punct:
        score += 0.1
    
    return min(score, 1.0)


def compute_completeness_score(text: str) -> float:
    """
    Compute completeness score based on length and detail indicators.
    
    Higher score = more complete prompt.
    Paper reference: Section 3.2
    """
    words = text.split()
    token_count = len(words)
    
    score = 1.0
    
    # Penalize very short prompts
    if token_count < 5:
        score -= 0.4
    elif token_count < 10:
        score -= 0.2
    
    # Penalize vague templates
    if re.search(r'(what is|tell me about)\s+\w+\s*\??$', text, re.IGNORECASE):
        score -= 0.15
    
    # Penalize single-word queries
    if token_count == 1:
        score -= 0.3
        
    return max(score, 0.0)


def compute_fluency_score(text: str) -> float:
    """
    Compute fluency score based on fragments and repetition.
    
    Higher score = more fluent text.
    Paper reference: Section 3.2
    """
    words = text.split()
    if len(words) < 3:
        return 0.5  # Short prompts are inherently less fluent
    
    score = 1.0
    
    # Check for repeated bigrams (sign of disfluency)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    unique_bigrams = set(bigrams)
    if len(unique_bigrams) < len(bigrams) * 0.9:  # >10% repetition
        score -= 0.2
    
    # Check for fragments (very short sentences)
    if len(words) < 3:
        score -= 0.3
        
    return max(score, 0.0)


def compute_clarity_score(text: str) -> float:
    """
    Compute clarity score based on lexical diversity and ambiguous references.
    
    Higher score = clearer text.
    Paper reference: Section 3.2
    """
    words = text.split()
    if len(words) < 5:
        return 1.0  # Short prompts are clear by default
    
    score = 1.0
    
    # Check lexical diversity (type-token ratio)
    if len(words) >= 12:
        ttr = len(set(w.lower() for w in words)) / len(words)
        if ttr < 0.4:
            score -= 0.2
    
    # Check for ambiguous pronouns at start
    if re.match(r'(it|this|that|they|them)\b', text.lower()):
        score -= 0.15
        
    return max(score, 0.0)


def compute_quality_score(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Compute overall quality score and component scores.
    
    Paper reference: Section 3.2, Eq. 3
    """
    typo_score = compute_typo_score(text)
    comp_score = compute_completeness_score(text)
    flu_score = compute_fluency_score(text)
    clar_score = compute_clarity_score(text)
    
    # Overall quality score (manuscript Eq. 3)
    # Quality decreases with high typo score or low other scores
    quality = 1 - max(
        typo_score,
        max(0, COMPLETENESS_THRESHOLD - comp_score),
        max(0, FLUENCY_THRESHOLD - flu_score),
        max(0, CLARITY_THRESHOLD - clar_score)
    )
    
    scores = {
        "typo": typo_score,
        "completeness": comp_score,
        "fluency": flu_score,
        "clarity": clar_score,
        "quality": quality
    }
    
    return quality, scores


def should_skip(quality_score: float, typo_score: float, skip_enabled: bool = True) -> bool:
    """
    Conservative skip logic.
    
    Paper reference: Section 3.2
    Skip if quality > 0.75 and typo < 0.20
    """
    if not skip_enabled:
        return False
    return quality_score > (1 - SKIP_THRESHOLD) and typo_score < 0.20


def select_workers(scores: Dict[str, float]) -> List[str]:
    """Select appropriate workers based on threshold routing."""
    workers = []
    
    if scores["typo"] > TYPO_THRESHOLD:
        workers.append("cleaner")
    if scores["completeness"] < COMPLETENESS_THRESHOLD:
        workers.append("fact_adder")
    if scores["fluency"] < FLUENCY_THRESHOLD:
        workers.append("paraphraser")
        
    return workers


def compute_drift(original: str, modified: str) -> float:
    """
    Compute lexical similarity drift.
    
    Paper reference: Section 3.5, Eq. 1
    Returns drift value (0 = identical, 1 = completely different)
    """
    # Simple lexical similarity using SequenceMatcher
    similarity = SequenceMatcher(None, original.lower(), modified.lower()).ratio()
    return 1.0 - similarity


def within_drift_bound(original: str, modified: str, drift_enabled: bool = True) -> bool:
    """Check if edit is within acceptable drift bound."""
    if not drift_enabled:
        return True
    return compute_drift(original, modified) <= MAX_DRIFT


def within_length_bound(original: str, modified: str, drift_enabled: bool = True) -> bool:
    """Check if edit is within acceptable length expansion."""
    if not drift_enabled:
        return True
    if not original:
        return True
    return len(modified) / len(original) <= MAX_LENGTH_RATIO


async def call_worker(url: str, endpoint: str, payload: Dict) -> Tuple[Dict, float]:
    """Call a specialist worker service."""
    start_time = time.perf_counter()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{url}/{endpoint}", json=payload)
        response.raise_for_status()
        result = response.json()
    
    latency = (time.perf_counter() - start_time) * 1000
    return result, latency


def merge_outputs(
    original: str,
    cleaned: Optional[str] = None,
    facts: Optional[str] = None,
    paraphrased: Optional[str] = None,
    workers_used: List[str] = None,
    drift_enabled: bool = True
) -> Tuple[str, List[str]]:
    """
    Merge specialist outputs with drift control.
    
    Paper reference: Section 3.5
    Order: Cleaner → Paraphraser → Facts (prepended)
    """
    if workers_used is None:
        workers_used = []
    
    # Start with original
    result = original
    applied_workers = []
    
    # Apply Cleaner first (if available and within bounds)
    if cleaned and "cleaner" in workers_used:
        if within_drift_bound(original, cleaned, drift_enabled) and \
           within_length_bound(original, cleaned, drift_enabled):
            result = cleaned
            applied_workers.append("cleaner")
    
    # Apply Paraphraser next (if available and within bounds)
    if paraphrased and "paraphraser" in workers_used:
        if within_drift_bound(original, paraphrased, drift_enabled) and \
           within_length_bound(original, paraphrased, drift_enabled):
            result = paraphrased
            applied_workers.append("paraphraser")
    
    # Prepend facts (if available and within bounds)
    if facts and facts.strip() and facts.upper() != "NONE" and "fact_adder" in workers_used:
        fact_lines = facts.strip().split('\n')[:MAX_FACTS]  # Limit to max facts
        
        # Verify facts are within token limit (approximate)
        total_fact_chars = sum(len(line) for line in fact_lines)
        if total_fact_chars <= FACT_TOKEN_LIMIT * 4:  # Rough chars-to-tokens conversion
            facts_text = '\n'.join(fact_lines) + '\n\n'
            merged = facts_text + result
            
            if within_length_bound(original, merged, drift_enabled):
                result = merged
                applied_workers.append("fact_adder")
    
    return result, applied_workers


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "POaaS Orchestrator",
        "version": "1.0.0",
        "ablation_mode": ABLATION_MODE,
        "config": {
            "typo_threshold": TYPO_THRESHOLD,
            "completeness_threshold": COMPLETENESS_THRESHOLD,
            "fluency_threshold": FLUENCY_THRESHOLD,
            "skip_threshold": SKIP_THRESHOLD,
            "max_drift": MAX_DRIFT,
            "max_length_ratio": MAX_LENGTH_RATIO
        }
    }


@app.post("/infer")
async def infer(request: InferRequest, req: Request = None) -> InferResponse:
    """
    Main inference endpoint for POaaS prompt optimization.
    
    Supports ablation modes via request body or environment variable.
    """
    start_time = time.perf_counter()
    prompt = request.prompt
    
    # Determine ablation mode
    ablation_mode = request.ablation or ABLATION_MODE
    skip_enabled, drift_enabled = get_ablation_flags(ablation_mode)
    
    # Extract run_id from headers if present
    run_id = None
    if req and hasattr(req, 'headers'):
        run_id = req.headers.get("X-Run-Id")
    
    # Phase 1: Prompt analysis and routing (Section 3.2)
    quality_score, component_scores = compute_quality_score(prompt)
    
    # Phase 2: Conservative skip logic
    if should_skip(quality_score, component_scores["typo"], skip_enabled):
        total_latency = (time.perf_counter() - start_time) * 1000
        return InferResponse(
            final_prompt=prompt,
            skipped=True,
            workers_used=[],
            latency_ms=total_latency,
            reasoning=f"High quality prompt (q={quality_score:.3f}) - conservative skip",
            ablation_mode=ablation_mode,
            run_id=run_id
        )
    
    # Phase 3: Worker selection
    workers_to_call = select_workers(component_scores)
    
    if not workers_to_call:
        total_latency = (time.perf_counter() - start_time) * 1000
        return InferResponse(
            final_prompt=prompt,
            skipped=True,
            workers_used=[],
            latency_ms=total_latency,
            reasoning="No workers needed based on thresholds",
            ablation_mode=ablation_mode,
            run_id=run_id
        )
    
    # Phase 4: Parallel worker calls
    worker_results = {}
    worker_latencies = {}
    
    tasks = []
    if "cleaner" in workers_to_call:
        tasks.append(("cleaner", call_worker(CLEANER_URL, "clean", {"text": prompt})))
    if "paraphraser" in workers_to_call:
        tasks.append(("paraphraser", call_worker(PARAPHRASER_URL, "paraphrase", {"text": prompt})))
    if "fact_adder" in workers_to_call:
        tasks.append(("fact_adder", call_worker(FACT_ADDER_URL, "facts", {"text": prompt})))
    
    # Execute all calls in parallel
    if tasks:
        try:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for i, (worker_name, _) in enumerate(tasks):
                if isinstance(results[i], Exception):
                    # On error, skip this worker
                    continue
                
                result_data, latency = results[i]
                worker_results[worker_name] = result_data
                worker_latencies[worker_name] = latency
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Worker call failed: {str(e)}")
    
    # Phase 5: Drift-controlled merging (Section 3.5)
    cleaned = worker_results.get("cleaner", {}).get("cleaned")
    facts = worker_results.get("fact_adder", {}).get("facts")
    paraphrased = worker_results.get("paraphraser", {}).get("paraphrased")
    
    final_prompt, applied_workers = merge_outputs(
        prompt, cleaned, facts, paraphrased, workers_to_call, drift_enabled
    )
    
    total_latency = (time.perf_counter() - start_time) * 1000
    
    return InferResponse(
        final_prompt=final_prompt,
        skipped=False,
        workers_used=applied_workers,
        latency_ms=total_latency,
        reasoning=f"Applied {len(applied_workers)}/{len(workers_to_call)} workers: {applied_workers}",
        ablation_mode=ablation_mode,
        run_id=run_id
    )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="POaaS Orchestrator")
    parser.add_argument("--port", type=int, default=8001, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--ablation", choices=["full", "no_skip", "no_drift"], 
                        default="full", help="Ablation mode")
    
    args = parser.parse_args()
    
    # Set ablation mode from CLI
    if args.ablation:
        os.environ["POAAS_ABLATION"] = args.ablation
    
    uvicorn.run(app, host=args.host, port=args.port)
