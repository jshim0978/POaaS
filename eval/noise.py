#!/usr/bin/env python3
"""
Noise injection for POaaS degradation experiments.

Implements the input degradation protocol from Appendix J:
- Token deletion: delete r ∈ {0.05, 0.10, 0.15} of word tokens uniformly at random
- Token mixup: replace r ∈ {0.05, 0.10, 0.15} of word tokens with content words

All perturbations are deterministic under a fixed seed.

Usage:
    from eval.noise import apply_token_deletion, apply_token_mixup, apply_noise
    
    # Apply 10% token deletion with seed 13
    degraded = apply_token_deletion(text, rate=0.10, seed=13)
    
    # Apply 15% token mixup with seed 13
    degraded = apply_token_mixup(text, rate=0.15, seed=13)
"""

import random
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    noise_type: str = "clean"  # "clean", "deletion", or "mixup"
    rate: float = 0.0  # 0.05, 0.10, or 0.15
    seed: int = 13
    
    @property
    def name(self) -> str:
        """Get human-readable name for this noise condition."""
        if self.noise_type == "clean":
            return "clean"
        elif self.noise_type == "deletion":
            return f"del-{int(self.rate * 100)}"
        elif self.noise_type == "mixup":
            return f"mix-{int(self.rate * 100)}"
        else:
            return f"{self.noise_type}-{self.rate}"
    
    def apply(self, text: str) -> str:
        """Apply this noise configuration to text."""
        return apply_noise(text, self.noise_type, self.rate, self.seed)

# Common content words for mixup (vocabulary for replacement)
MIXUP_VOCABULARY = [
    # Nouns
    "apple", "book", "car", "dog", "elephant", "flower", "garden", "house",
    "island", "jacket", "kitchen", "lamp", "mountain", "notebook", "ocean",
    "pencil", "question", "river", "stone", "table", "umbrella", "village",
    "window", "xylophone", "yellow", "zebra", "computer", "science", "world",
    "system", "problem", "solution", "method", "result", "study", "research",
    "data", "model", "process", "theory", "example", "case", "point", "fact",
    "number", "time", "year", "day", "way", "thing", "man", "woman", "child",
    "part", "place", "work", "week", "company", "group", "country", "hand",
    # Verbs
    "run", "jump", "think", "believe", "create", "develop", "find", "give",
    "help", "include", "keep", "lead", "make", "need", "offer", "provide",
    "require", "see", "take", "use", "want", "work", "allow", "become",
    # Adjectives
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next",
    "early", "young", "important", "few", "public", "bad", "same", "able",
    # Adverbs
    "also", "just", "only", "now", "very", "well", "even", "still", "already",
    "often", "always", "never", "really", "quite", "rather", "almost",
]


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words, preserving punctuation as separate tokens.
    
    This matches the paper's definition of "word tokens" for perturbation.
    """
    # Split on whitespace, keeping punctuation attached for now
    tokens = text.split()
    return tokens


def detokenize(tokens: List[str]) -> str:
    """Reconstruct text from tokens."""
    return " ".join(tokens)


def apply_token_deletion(text: str, rate: float, seed: int) -> str:
    """
    Apply token deletion to text.
    
    Deletes r fraction of word tokens uniformly at random.
    Implements Appendix J: "delete r ∈ {0.05, 0.10, 0.15} of word tokens uniformly at random"
    
    Args:
        text: Input text to perturb
        rate: Deletion rate (e.g., 0.05, 0.10, 0.15)
        seed: Random seed for deterministic perturbation
        
    Returns:
        Perturbed text with tokens deleted
    """
    if not text or rate <= 0:
        return text
    
    random.seed(seed)
    tokens = tokenize(text)
    
    if len(tokens) <= 1:
        return text
    
    # Calculate number of tokens to delete
    n_delete = max(1, int(len(tokens) * rate))
    n_delete = min(n_delete, len(tokens) - 1)  # Keep at least one token
    
    # Select indices to delete
    indices_to_delete = set(random.sample(range(len(tokens)), n_delete))
    
    # Keep tokens not in deletion set
    result_tokens = [tok for i, tok in enumerate(tokens) if i not in indices_to_delete]
    
    return detokenize(result_tokens)


def apply_token_mixup(text: str, rate: float, seed: int, vocabulary: Optional[List[str]] = None) -> str:
    """
    Apply token mixup (replacement) to text.
    
    Replaces r fraction of word tokens with content words sampled from vocabulary.
    Implements Appendix J: "replace r ∈ {0.05, 0.10, 0.15} of word tokens with 
    content words sampled from a fixed vocabulary"
    
    Args:
        text: Input text to perturb
        rate: Replacement rate (e.g., 0.05, 0.10, 0.15)
        seed: Random seed for deterministic perturbation
        vocabulary: Optional custom vocabulary (defaults to MIXUP_VOCABULARY)
        
    Returns:
        Perturbed text with tokens replaced
    """
    if not text or rate <= 0:
        return text
    
    if vocabulary is None:
        vocabulary = MIXUP_VOCABULARY
    
    random.seed(seed)
    tokens = tokenize(text)
    
    if len(tokens) <= 1:
        return text
    
    # Calculate number of tokens to replace
    n_replace = max(1, int(len(tokens) * rate))
    n_replace = min(n_replace, len(tokens))
    
    # Select indices to replace
    indices_to_replace = set(random.sample(range(len(tokens)), n_replace))
    
    # Replace selected tokens with random vocabulary words
    result_tokens = []
    for i, tok in enumerate(tokens):
        if i in indices_to_replace:
            # Sample replacement word
            replacement = random.choice(vocabulary)
            # Try to preserve case
            if tok.isupper():
                replacement = replacement.upper()
            elif tok[0].isupper() if tok else False:
                replacement = replacement.capitalize()
            result_tokens.append(replacement)
        else:
            result_tokens.append(tok)
    
    return detokenize(result_tokens)


def apply_noise(text: str, noise_type: str, rate: float, seed: int) -> str:
    """
    Apply noise to text based on type and rate.
    
    Args:
        text: Input text
        noise_type: One of "clean", "deletion", "mixup"
        rate: Noise rate (0.0 for clean, 0.05/0.10/0.15 for others)
        seed: Random seed
        
    Returns:
        Perturbed (or original if clean) text
    """
    if noise_type == "clean" or rate <= 0:
        return text
    elif noise_type == "deletion":
        return apply_token_deletion(text, rate, seed)
    elif noise_type == "mixup":
        return apply_token_mixup(text, rate, seed)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def get_noise_conditions() -> List[dict]:
    """
    Get all noise conditions from the paper.
    
    Returns list of condition dicts with 'name', 'type', and 'rate'.
    """
    return [
        {"name": "clean", "type": "clean", "rate": 0.0},
        {"name": "del-5", "type": "deletion", "rate": 0.05},
        {"name": "del-10", "type": "deletion", "rate": 0.10},
        {"name": "del-15", "type": "deletion", "rate": 0.15},
        {"name": "mix-5", "type": "mixup", "rate": 0.05},
        {"name": "mix-10", "type": "mixup", "rate": 0.10},
        {"name": "mix-15", "type": "mixup", "rate": 0.15},
    ]


# Convenience functions for common noise levels
def apply_deletion_5(text: str, seed: int = 13) -> str:
    """Apply 5% token deletion."""
    return apply_token_deletion(text, 0.05, seed)


def apply_deletion_10(text: str, seed: int = 13) -> str:
    """Apply 10% token deletion."""
    return apply_token_deletion(text, 0.10, seed)


def apply_deletion_15(text: str, seed: int = 13) -> str:
    """Apply 15% token deletion."""
    return apply_token_deletion(text, 0.15, seed)


def apply_mixup_5(text: str, seed: int = 13) -> str:
    """Apply 5% token mixup."""
    return apply_token_mixup(text, 0.05, seed)


def apply_mixup_10(text: str, seed: int = 13) -> str:
    """Apply 10% token mixup."""
    return apply_token_mixup(text, 0.10, seed)


def apply_mixup_15(text: str, seed: int = 13) -> str:
    """Apply 15% token mixup."""
    return apply_token_mixup(text, 0.15, seed)


if __name__ == "__main__":
    # Test the noise functions
    test_text = "What is the capital of France? Please provide a detailed answer."
    
    print("Original:", test_text)
    print()
    
    for condition in get_noise_conditions():
        if condition["type"] == "clean":
            result = test_text
        else:
            result = apply_noise(test_text, condition["type"], condition["rate"], seed=13)
        print(f"{condition['name']:>8}: {result}")

