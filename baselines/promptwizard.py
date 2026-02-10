"""
Real PromptWizard implementation for POaaS baseline comparison.

Based on: "PromptWizard: Task-Aware Agent-driven Prompt Optimization Framework"
https://arxiv.org/abs/2405.12447

Implements PromptWizard algorithm with:
- Task-aware prompt analysis
- Multi-agent critique and synthesis
- Iterative refinement process
- Knowledge incorporation
"""

import asyncio
import random
from typing import List, Dict, Any, Tuple
from poaas.common.vllm_client import chat
from baselines.async_utils import run_async


class PromptWizardBaseline:
    """Real PromptWizard implementation using task-aware agent-driven optimization."""
    
    def __init__(self, model="meta-llama/Llama-3.2-3B-Instruct", num_rounds=3, num_candidates=4):
        self.model = model
        self.num_rounds = num_rounds
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
        Optimize prompt using critique-and-synthesis approach.
        
        Args:
            initial_prompt: Starting prompt
            task_examples: Training examples for optimization
            
        Returns:
            Optimized prompt string
        """
        current_prompt = initial_prompt
        
        for round_num in range(self.num_rounds):
            print(f"PromptWizard Round {round_num + 1}/{self.num_rounds}")
            
            # Step 1: Analyze task requirements
            task_analysis = await self._analyze_task(task_examples)
            
            # Step 2: Generate candidate instructions with different strategies
            candidates = await self._generate_candidates(current_prompt, task_examples, task_analysis)
            
            # Step 3: Critique each candidate
            critiques = await self._critique_candidates(candidates, task_examples, task_analysis)
            
            # Step 4: Synthesize improved prompt
            current_prompt = await self._synthesize_prompt(candidates, critiques, task_examples, task_analysis)
            
            print(f"Round {round_num + 1} result: {current_prompt[:100]}...")
        
        return current_prompt
    
    async def _analyze_task(self, task_examples: List[Dict]) -> Dict[str, Any]:
        """Analyze task characteristics and requirements."""
        if not task_examples:
            return {
                "task_type": "general",
                "complexity": "medium",
                "requirements": ["clarity", "accuracy"]
            }
        
        try:
            # Sample a few examples for analysis
            sample_examples = task_examples[:3]
            examples_text = ""
            
            for i, example in enumerate(sample_examples):
                if "input" in example and "output" in example:
                    examples_text += f"Example {i+1}:\nInput: {example['input']}\nOutput: {example['output']}\n\n"
                elif "question" in example and "answer" in example:
                    examples_text += f"Example {i+1}:\nQuestion: {example['question']}\nAnswer: {example['answer']}\n\n"
            
            analysis_prompt = (
                f"Analyze the following task examples and identify key characteristics:\n\n"
                f"{examples_text}"
                f"Based on these examples, provide:\n"
                f"1. Task type (reasoning, math, factual, creative, etc.)\n"
                f"2. Complexity level (simple, medium, complex)\n"
                f"3. Key requirements for good prompts\n"
                f"4. Common patterns in the examples\n\n"
                f"Respond in a structured format."
            )
            
            result = await chat([
                {"role": "system", "content": "You are a task analysis expert. Analyze tasks and identify optimization requirements."},
                {"role": "user", "content": analysis_prompt}
            ], temperature=0.3, max_tokens=200, model=self.model)
            
            analysis_text = result["text"].strip()
            
            # Parse analysis (simplified)
            task_analysis = {
                "task_type": self._extract_task_type(analysis_text),
                "complexity": self._extract_complexity(analysis_text),
                "requirements": self._extract_requirements(analysis_text),
                "analysis_text": analysis_text
            }
            
        except Exception as e:
            print(f"Task analysis error: {e}")
            # Fallback analysis
            task_analysis = {
                "task_type": "reasoning",
                "complexity": "medium", 
                "requirements": ["step-by-step thinking", "accuracy", "clarity"],
                "analysis_text": "Fallback analysis: reasoning task requiring careful thought."
            }
        
        return task_analysis
    
    def _extract_task_type(self, analysis: str) -> str:
        """Extract task type from analysis text."""
        analysis_lower = analysis.lower()
        
        if any(word in analysis_lower for word in ["math", "arithmetic", "calculation"]):
            return "mathematics"
        elif any(word in analysis_lower for word in ["reasoning", "logic", "problem"]):
            return "reasoning"
        elif any(word in analysis_lower for word in ["factual", "knowledge", "information"]):
            return "factual"
        elif any(word in analysis_lower for word in ["creative", "generate", "write"]):
            return "creative"
        else:
            return "general"
    
    def _extract_complexity(self, analysis: str) -> str:
        """Extract complexity level from analysis text."""
        analysis_lower = analysis.lower()
        
        if any(word in analysis_lower for word in ["simple", "easy", "basic"]):
            return "simple"
        elif any(word in analysis_lower for word in ["complex", "difficult", "advanced"]):
            return "complex"
        else:
            return "medium"
    
    def _extract_requirements(self, analysis: str) -> List[str]:
        """Extract key requirements from analysis text."""
        requirements = []
        analysis_lower = analysis.lower()
        
        requirement_keywords = {
            "step": "step-by-step reasoning",
            "precise": "precision",
            "accurate": "accuracy", 
            "clear": "clarity",
            "detail": "detailed explanation",
            "example": "examples",
            "structure": "structured approach"
        }
        
        for keyword, requirement in requirement_keywords.items():
            if keyword in analysis_lower:
                requirements.append(requirement)
        
        return requirements if requirements else ["clarity", "accuracy"]
    
    async def _generate_candidates(self, prompt: str, task_examples: List[Dict], task_analysis: Dict) -> List[str]:
        """Generate multiple candidate prompt variations using different strategies."""
        candidates = []
        
        # Strategy 1: Structure-based improvement
        candidates.append(await self._add_structure_strategy(prompt, task_analysis))
        
        # Strategy 2: Task-specific enhancement
        candidates.append(await self._task_specific_strategy(prompt, task_analysis))
        
        # Strategy 3: Example incorporation
        candidates.append(await self._example_strategy(prompt, task_examples))
        
        # Strategy 4: Reasoning enhancement
        candidates.append(await self._reasoning_strategy(prompt, task_analysis))
        
        # Fill remaining slots if needed
        while len(candidates) < self.num_candidates:
            candidates.append(await self._general_improvement_strategy(prompt, len(candidates)))
        
        # Remove duplicates and filter quality
        unique_candidates = []
        for candidate in candidates:
            if candidate and len(candidate) > 10 and candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        return unique_candidates[:self.num_candidates]
    
    async def _add_structure_strategy(self, prompt: str, task_analysis: Dict) -> str:
        """Add structure to the prompt based on task analysis."""
        try:
            structure_prompt = (
                f"Add clear structure and organization to this prompt based on the task type '{task_analysis['task_type']}':\n\n"
                f"{prompt}\n\n"
                f"Make it more structured and step-by-step without changing the core instruction."
            )
            
            result = await chat([
                {"role": "system", "content": "You are a prompt structure expert. Add helpful organization to prompts."},
                {"role": "user", "content": structure_prompt}
            ], temperature=0.4, max_tokens=200, model=self.model)
            
            return result["text"].strip()
            
        except:
            # Fallback structure improvement
            if task_analysis["task_type"] == "mathematics":
                return f"Solve this step by step:\n1. Read the problem carefully\n2. Identify what you need to find\n3. Work through the solution\n4. Check your answer\n\n{prompt}"
            else:
                return f"Please approach this systematically:\n{prompt}\n\nThink through each step carefully."
    
    async def _task_specific_strategy(self, prompt: str, task_analysis: Dict) -> str:
        """Enhance prompt with task-specific improvements."""
        try:
            task_type = task_analysis["task_type"]
            requirements = ", ".join(task_analysis["requirements"])
            
            enhancement_prompt = (
                f"Improve this prompt for a {task_type} task that requires {requirements}:\n\n"
                f"{prompt}\n\n"
                f"Add task-specific guidance and requirements while keeping it concise."
            )
            
            result = await chat([
                {"role": "system", "content": f"You are an expert in {task_type} tasks. Optimize prompts for this domain."},
                {"role": "user", "content": enhancement_prompt}
            ], temperature=0.5, max_tokens=180, model=self.model)
            
            return result["text"].strip()
            
        except:
            # Fallback task-specific improvements
            task_specific_additions = {
                "mathematics": " Show your work step by step and double-check your calculations.",
                "reasoning": " Think through this logically and consider all relevant factors.",
                "factual": " Provide accurate and verifiable information.",
                "creative": " Be creative and original in your response."
            }
            
            addition = task_specific_additions.get(task_analysis["task_type"], " Be thorough and accurate.")
            return prompt + addition
    
    async def _example_strategy(self, prompt: str, task_examples: List[Dict]) -> str:
        """Add relevant examples to the prompt."""
        if not task_examples:
            return prompt
        
        try:
            # Select a good example to include
            example = task_examples[0] if task_examples else None
            if not example:
                return prompt
            
            example_prompt = (
                f"Add a helpful example to this prompt to make it clearer:\n\n"
                f"{prompt}\n\n"
                f"Use this as inspiration for the example format:\n"
            )
            
            if "input" in example and "output" in example:
                example_prompt += f"Input: {example['input']}\nOutput: {example['output']}"
            elif "question" in example and "answer" in example:
                example_prompt += f"Question: {example['question']}\nAnswer: {example['answer']}"
            
            result = await chat([
                {"role": "system", "content": "You are a prompt enhancement expert. Add helpful examples to prompts."},
                {"role": "user", "content": example_prompt}
            ], temperature=0.6, max_tokens=220, model=self.model)
            
            return result["text"].strip()
            
        except:
            # Fallback: simple example addition
            if task_examples and "question" in task_examples[0]:
                return f"{prompt}\n\nFor example: {task_examples[0]['question']}"
            else:
                return f"{prompt}\n\nHere's how to approach this type of problem..."
    
    async def _reasoning_strategy(self, prompt: str, task_analysis: Dict) -> str:
        """Enhance reasoning capabilities in the prompt."""
        try:
            reasoning_prompt = (
                f"Add reasoning and thinking guidance to this prompt for {task_analysis['complexity']} level tasks:\n\n"
                f"{prompt}\n\n"
                f"Help the model think through the problem systematically."
            )
            
            result = await chat([
                {"role": "system", "content": "You are a reasoning expert. Add thinking guidance to prompts."},
                {"role": "user", "content": reasoning_prompt}
            ], temperature=0.4, max_tokens=190, model=self.model)
            
            return result["text"].strip()
            
        except:
            # Fallback reasoning enhancement
            return f"{prompt}\n\nLet's think about this step by step:\n1. What information do we have?\n2. What do we need to find?\n3. How can we solve this?"
    
    async def _general_improvement_strategy(self, prompt: str, variant_id: int) -> str:
        """General prompt improvement strategies."""
        strategies = [
            f"Please {prompt.lower().lstrip('please ')} Be clear and precise.",
            f"Carefully consider: {prompt}",
            f"Think through this problem: {prompt}",
            f"Your task: {prompt} Explain your reasoning.",
            f"Solve this systematically: {prompt}"
        ]
        
        return strategies[variant_id % len(strategies)]
    
    async def _critique_candidates(self, candidates: List[str], task_examples: List[Dict], task_analysis: Dict) -> List[str]:
        """Critique each candidate prompt for strengths and weaknesses."""
        critiques = []
        
        for candidate in candidates:
            try:
                critique_prompt = (
                    f"Critique this prompt for a {task_analysis['task_type']} task:\n\n"
                    f"{candidate}\n\n"
                    f"Task requirements: {', '.join(task_analysis['requirements'])}\n\n"
                    f"Identify:\n"
                    f"1. Strengths of this prompt\n"
                    f"2. Potential weaknesses or missing elements\n"
                    f"3. Suggestions for improvement\n\n"
                    f"Be specific and actionable."
                )
                
                result = await chat([
                    {"role": "system", "content": "You are a prompt evaluation expert. Provide constructive criticism."},
                    {"role": "user", "content": critique_prompt}
                ], temperature=0.3, max_tokens=150, model=self.model)
                
                critiques.append(result["text"].strip())
                
            except Exception as e:
                print(f"Critique generation error: {e}")
                # Fallback critique
                critiques.append(f"This prompt could benefit from more specific instructions and clearer structure.")
        
        return critiques
    
    async def _synthesize_prompt(self, candidates: List[str], critiques: List[str], task_examples: List[Dict], task_analysis: Dict) -> str:
        """Synthesize improved prompt based on candidates and critiques."""
        try:
            synthesis_content = f"Task type: {task_analysis['task_type']}\n"
            synthesis_content += f"Requirements: {', '.join(task_analysis['requirements'])}\n\n"
            
            synthesis_content += "Candidate prompts and their critiques:\n\n"
            
            for i, (candidate, critique) in enumerate(zip(candidates, critiques)):
                synthesis_content += f"Candidate {i+1}: {candidate}\n"
                synthesis_content += f"Critique {i+1}: {critique}\n\n"
            
            synthesis_content += (
                "Based on the candidates and critiques above, create ONE improved prompt that:\n"
                "1. Incorporates the best elements from all candidates\n"
                "2. Addresses the weaknesses identified in critiques\n"
                "3. Meets the task requirements effectively\n"
                "4. Is clear, concise, and actionable\n\n"
                "Provide only the final improved prompt:"
            )
            
            result = await chat([
                {"role": "system", "content": "You are a prompt synthesis expert. Combine the best elements to create optimal prompts."},
                {"role": "user", "content": synthesis_content}
            ], temperature=0.4, max_tokens=250, model=self.model)
            
            synthesized = result["text"].strip()
            
            # Clean up the result
            synthesized = self._extract_prompt_from_response(synthesized)
            
            return synthesized if len(synthesized) > 10 else candidates[0] if candidates else "Please solve this problem."
            
        except Exception as e:
            print(f"Synthesis error: {e}")
            # Fallback: return best candidate or simple combination
            if candidates:
                return candidates[0]  # Return first candidate as fallback
            else:
                return "Please solve this problem step by step."
    
    def _extract_prompt_from_response(self, response: str) -> str:
        """Extract the actual prompt from synthesis response."""
        # Remove common prefixes and suffixes
        prefixes = [
            "improved prompt:", "final prompt:", "synthesized prompt:", 
            "here's the improved prompt:", "the improved prompt is:"
        ]
        
        cleaned = response.strip()
        
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        # Take first substantial line if multiline
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        if lines:
            # Return longest line (likely the main prompt)
            return max(lines, key=len)
        
        return cleaned


async def optimize_with_promptwizard(
    initial_prompt: str,
    task_examples: List[Dict],
    num_rounds: int = 3,
    num_candidates: int = 4
) -> str:
    """
    Optimize a prompt using PromptWizard algorithm.
    
    Args:
        initial_prompt: Starting prompt
        task_examples: Training examples as [{"input": str, "output": str}, ...]
        num_rounds: Number of critique-synthesis rounds
        num_candidates: Number of candidate prompts per round
    
    Returns:
        Optimized prompt
    """
    optimizer = PromptWizardBaseline(
        num_rounds=num_rounds,
        num_candidates=num_candidates
    )
    
    return await optimizer._optimize_prompt_async(initial_prompt, task_examples)


if __name__ == "__main__":
    # Test the implementation
    async def test():
        initial_prompt = "Solve this problem:"
        examples = [
            {"input": "If a train travels 60 mph for 2 hours, how far does it go?", "output": "120 miles"},
            {"input": "What is 15% of 80?", "output": "12"}
        ]
        
        optimized = await optimize_with_promptwizard(initial_prompt, examples, 2, 3)
        print(f"Optimized prompt: {optimized}")
    
    asyncio.run(test())