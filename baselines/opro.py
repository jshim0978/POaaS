"""
Real OPRO (Optimization by PROmpting) implementation for POaaS baseline comparison.

Based on: "Large Language Models as Optimizers"
https://arxiv.org/abs/2309.03409

Implements OPRO algorithm with:
- LLM as meta-optimizer
- Iterative prompt improvement
- Performance-based optimization history
- Meta-prompt engineering
"""

import asyncio
import random
from typing import List, Dict, Any, Tuple
from poaas.common.vllm_client import chat
from baselines.async_utils import run_async


class OPROBaseline:
    """Real OPRO implementation using LLM as optimizer."""
    
    def __init__(self, model="meta-llama/Llama-3.2-3B-Instruct", max_iterations=8, num_candidates=5):
        self.model = model
        self.max_iterations = max_iterations
        self.num_candidates = num_candidates
        
    def optimize_prompt(self, initial_prompt, task_examples=None):
        """Synchronous wrapper for async optimize_prompt method."""
        if task_examples is None:
            task_examples = []
        
        # Convert to expected format if needed
        formatted_examples = []
        for example in task_examples:
            if isinstance(example, dict):
                formatted_examples.append(example)
            else:
                formatted_examples.append({"input": str(example), "output": "sample_output"})
        
        # Run async optimization
        return run_async(self._optimize_prompt_async(initial_prompt, formatted_examples))
        
    async def _optimize_prompt_async(self, initial_prompt: str, task_examples: List[Dict]) -> str:
        """
        Optimize prompt using LLM as meta-optimizer.
        
        Args:
            initial_prompt: Starting prompt
            task_examples: Training examples for optimization
            
        Returns:
            Best optimized prompt
        """
        current_prompt = initial_prompt
        best_score = await self._evaluate_prompt(current_prompt, task_examples)
        best_prompt = current_prompt
        
        # Track optimization history
        optimization_history = [(current_prompt, best_score)]
        
        for iteration in range(self.max_iterations):
            print(f"OPRO Iteration {iteration + 1}: Current best score = {best_score:.3f}")
            
            # Generate candidate prompts using meta-prompt
            candidates = await self._generate_candidates(optimization_history, task_examples)
            
            # Evaluate all candidates
            candidate_scores = []
            for candidate in candidates:
                score = await self._evaluate_prompt(candidate, task_examples)
                candidate_scores.append((candidate, score))
                optimization_history.append((candidate, score))
            
            # Select best candidate
            best_candidate, best_candidate_score = max(candidate_scores, key=lambda x: x[1])
            
            # Update best if improved
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_prompt = best_candidate
                current_prompt = best_candidate
            else:
                # Occasionally explore even if not better (with small probability)
                if random.random() < 0.2:
                    current_prompt = random.choice([c for c, s in candidate_scores])
        
        return best_prompt
    
    async def _generate_candidates(self, optimization_history: List[Tuple[str, float]], 
                                 task_examples: List[Dict]) -> List[str]:
        """Generate candidate prompts using meta-prompt optimization."""
        # Create meta-prompt with optimization history
        meta_prompt = self._create_meta_prompt(optimization_history, task_examples)
        
        candidates = []
        
        # Generate multiple candidates
        for i in range(self.num_candidates):
            try:
                result = await chat([
                    {"role": "system", "content": "You are an expert prompt optimizer. Generate improved prompts based on performance history."},
                    {"role": "user", "content": meta_prompt}
                ], temperature=0.7 + i * 0.1, max_tokens=200, model=self.model)  # Vary temperature for diversity
                
                candidate = result["text"].strip()
                
                # Extract just the prompt if wrapped in explanation
                candidate = self._extract_prompt_from_response(candidate)
                
                if len(candidate) > 10 and candidate not in [h[0] for h in optimization_history]:
                    candidates.append(candidate)
                    
            except Exception as e:
                print(f"Candidate generation error: {e}")
                # Fallback to rule-based generation
                if optimization_history:
                    base_prompt = optimization_history[-1][0]  # Use most recent
                    candidates.append(self._rule_based_improvement(base_prompt, i))
        
        # Ensure we have enough candidates
        while len(candidates) < self.num_candidates:
            if optimization_history:
                base_prompt = max(optimization_history, key=lambda x: x[1])[0]  # Use best
                candidates.append(self._rule_based_improvement(base_prompt, len(candidates)))
            else:
                candidates.append("Please solve this step by step.")
        
        return candidates[:self.num_candidates]
    
    def _create_meta_prompt(self, optimization_history: List[Tuple[str, float]], 
                           task_examples: List[Dict]) -> str:
        """Create meta-prompt for LLM-based optimization."""
        
        # Sort history by performance
        sorted_history = sorted(optimization_history, key=lambda x: x[1], reverse=True)
        
        # Show task context
        if task_examples:
            task_context = "Task examples:\n"
            for i, example in enumerate(task_examples[:3]):  # Show first 3
                if "input" in example and "output" in example:
                    task_context += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
                elif "question" in example and "answer" in example:
                    task_context += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
        else:
            task_context = "General prompt optimization task.\n\n"
        
        # Show performance history
        history_text = "Prompt optimization history (sorted by performance):\n\n"
        
        for i, (prompt, score) in enumerate(sorted_history[:8]):  # Show top 8
            history_text += f"Prompt {i+1} (Score: {score:.3f}): {prompt}\n\n"
        
        # Generate new prompt instruction
        instruction = (
            "Based on the task examples and the prompt performance history above, "
            "generate ONE new improved prompt that will likely achieve higher performance. "
            "The new prompt should:\n"
            "1. Learn from what works well in high-scoring prompts\n"
            "2. Avoid patterns from low-scoring prompts\n"
            "3. Be clear, specific, and well-structured\n"
            "4. Be different from existing prompts to explore new approaches\n\n"
            "Generate only the new prompt, without explanation:"
        )
        
        return task_context + history_text + instruction
    
    def _extract_prompt_from_response(self, response: str) -> str:
        """Extract the actual prompt from LLM response that might contain explanations."""
        # Look for common patterns where the prompt might be wrapped
        lines = response.split('\n')
        
        # Try to find the longest line (likely the prompt)
        prompt_candidates = [line.strip() for line in lines if len(line.strip()) > 20]
        
        if prompt_candidates:
            # Return the longest candidate (likely the actual prompt)
            return max(prompt_candidates, key=len)
        
        # Fallback: clean up the response
        cleaned = response.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "New prompt:", "Prompt:", "Improved prompt:", "Here's the new prompt:",
            "The new prompt is:", "Generated prompt:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        return cleaned
    
    def _rule_based_improvement(self, base_prompt: str, variant_id: int) -> str:
        """Generate rule-based improvements when LLM generation fails."""
        improvements = {
            0: f"Step by step: {base_prompt}",
            1: f"Think carefully and {base_prompt.lower()}",
            2: f"Let's work through this systematically: {base_prompt}",
            3: f"Please {base_prompt.lower().lstrip('please ')} Be precise.",
            4: f"Analyze the problem: {base_prompt}",
            5: f"Here's how to approach this: {base_prompt}",
            6: f"Consider all aspects: {base_prompt}"
        }
        
        return improvements.get(variant_id % 7, base_prompt)
    
    async def _evaluate_prompt(self, prompt: str, task_examples: List[Dict]) -> float:
        """Evaluate prompt performance on task examples."""
        if not task_examples:
            return 0.5  # Neutral score
        
        correct = 0
        total = min(len(task_examples), 4)  # Limit for efficiency
        
        for example in task_examples[:total]:
            try:
                # Format prompt with task input
                if "input" in example and "output" in example:
                    formatted_prompt = f"{prompt}\n\nInput: {example['input']}\nOutput:"
                    expected = example["output"]
                elif "question" in example and "answer" in example:
                    formatted_prompt = f"{prompt}\n\n{example['question']}"
                    expected = example["answer"]
                else:
                    continue
                
                # Get model response
                result = await chat([
                    {"role": "user", "content": formatted_prompt}
                ], temperature=0.1, max_tokens=100, model=self.model)
                
                response = result["text"].strip()
                
                # Check correctness
                if self._check_correctness(response, expected):
                    correct += 1
                    
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
        
        return correct / total if total > 0 else 0.0
    
    def _check_correctness(self, response: str, expected: str) -> bool:
        """Check if response matches expected output."""
        import re
        
        response_clean = re.sub(r'[^\w\s]', '', response.lower().strip())
        expected_clean = re.sub(r'[^\w\s]', '', expected.lower().strip())
        
        # Exact substring match
        if expected_clean in response_clean:
            return True
        
        # Numerical answers
        response_nums = re.findall(r'\d+(?:\.\d+)?', response)
        expected_nums = re.findall(r'\d+(?:\.\d+)?', expected)
        if response_nums and expected_nums:
            return response_nums[0] == expected_nums[0]
        
        # Multiple choice
        response_choices = re.findall(r'\b[ABCD]\b', response.upper())
        expected_choices = re.findall(r'\b[ABCD]\b', expected.upper())
        if response_choices and expected_choices:
            return response_choices[0] == expected_choices[0]
        
        return False


async def optimize_with_opro(
    initial_prompt: str,
    task_examples: List[Dict],
    max_iterations: int = 6,
    num_candidates: int = 4
) -> str:
    """
    Optimize a prompt using OPRO algorithm.
    
    Args:
        initial_prompt: Starting prompt
        task_examples: Training examples as [{"input": str, "output": str}, ...]
        max_iterations: Number of optimization iterations
        num_candidates: Number of candidate prompts per iteration
    
    Returns:
        Optimized prompt
    """
    optimizer = OPROBaseline(
        max_iterations=max_iterations,
        num_candidates=num_candidates
    )
    
    return await optimizer._optimize_prompt_async(initial_prompt, task_examples)


if __name__ == "__main__":
    # Test the implementation
    async def test():
        initial_prompt = "Answer this math question:"
        examples = [
            {"input": "What is 12 + 28?", "output": "40"},
            {"input": "What is 9 × 7?", "output": "63"}
        ]
        
        optimized = await optimize_with_opro(initial_prompt, examples, 3, 3)
        print(f"Optimized prompt: {optimized}")
    
    asyncio.run(test())