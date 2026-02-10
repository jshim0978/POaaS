"""
Real EvoPrompt implementation for POaaS baseline comparison.

Based on: "Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers"
https://arxiv.org/abs/2309.08532

Implements evolutionary prompt optimization with:
- Population-based evolution
- Differential evolution operators  
- Fitness evaluation on task examples
- LLM-guided mutations and crossover
"""

import asyncio
import random
import re
from typing import List, Dict, Any, Tuple
from poaas.common.vllm_client import chat
from baselines.async_utils import run_async


class EvoPromptBaseline:
    """Real EvoPrompt implementation using evolutionary algorithms."""
    
    def __init__(self, model="meta-llama/Llama-3.2-3B-Instruct", population_size=8, generations=10, mutation_rate=0.3):
        self.model = model
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = 0.7
        
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
                # Handle string examples
                formatted_examples.append({"input": str(example), "output": "sample_output"})
        
        # Run async optimization
        return run_async(self._optimize_prompt_async(initial_prompt, formatted_examples))
        
    async def _optimize_prompt_async(self, initial_prompt: str, task_examples: List[Dict]) -> str:
        """
        Optimize prompt using evolutionary algorithm.
        
        Args:
            initial_prompt: Starting prompt
            task_examples: Training examples for optimization
            
        Returns:
            Optimized prompt string
        """
        # Initialize population with diverse variations
        population = await self._initialize_population(initial_prompt)
        
        best_fitness = -float('inf')
        best_prompt = initial_prompt
        
        for generation in range(self.generations):
            # Evaluate fitness of all prompts on task examples
            fitness_scores = await self._evaluate_population(population, task_examples)
            
            # Track globally best prompt
            max_fitness_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_prompt = population[max_fitness_idx]
            
            # Create next generation through evolution
            population = await self._evolve_population(population, fitness_scores)
            
            print(f"EvoPrompt Gen {generation + 1}: Best fitness = {best_fitness:.3f}")
        
        return best_prompt
    
    async def _initialize_population(self, initial_prompt: str) -> List[str]:
        """Initialize population with diverse prompt variations using LLM."""
        population = [initial_prompt]  # Keep original
        
        # Generate semantic variations using LLM
        variation_instructions = [
            "Rephrase this prompt to be more specific and detailed:",
            "Simplify this prompt while keeping the core instruction:", 
            "Add step-by-step structure to this prompt:",
            "Make this prompt more conversational and engaging:",
            "Add helpful examples or context to this prompt:",
            "Restructure this prompt for better clarity:",
            "Make this prompt more concise but complete:"
        ]
        
        # Generate variations
        for i, instruction in enumerate(variation_instructions):
            if len(population) >= self.population_size:
                break
                
            try:
                result = await chat([
                    {"role": "system", "content": "You are a prompt optimization expert. Create improved prompt variations that maintain the original intent."},
                    {"role": "user", "content": f"{instruction}\n\n{initial_prompt}"}
                ], temperature=0.8, max_tokens=200, model=self.model)
                
                new_prompt = result["text"].strip()
                if len(new_prompt) > 10 and new_prompt != initial_prompt:
                    population.append(new_prompt)
            except:
                # Fallback to rule-based variation
                population.append(self._rule_based_variation(initial_prompt, i))
        
        # Fill remaining slots with mutations
        while len(population) < self.population_size:
            base = random.choice(population[:3])  # Use good prompts as base
            mutated = await self._mutate_prompt(base)
            population.append(mutated)
        
        return population[:self.population_size]
    
    def _rule_based_variation(self, prompt: str, variant_id: int) -> str:
        """Generate rule-based variations when LLM unavailable."""
        variations = {
            0: f"Please {prompt.lower().lstrip('please ')}",
            1: f"Step by step: {prompt}",
            2: f"Think carefully and {prompt.lower()}",
            3: f"Let's work through this: {prompt}", 
            4: f"Here's the task: {prompt}",
            5: f"Your goal is to {prompt.lower()}",
            6: f"Carefully {prompt.lower()}"
        }
        return variations.get(variant_id % 7, prompt)
    
    async def _evaluate_population(self, population: List[str], task_examples: List[Dict]) -> List[float]:
        """Evaluate fitness of each prompt on task examples."""
        fitness_scores = []
        
        # Limit examples for efficiency during optimization
        eval_examples = task_examples[:min(3, len(task_examples))]
        
        for prompt in population:
            score = await self._evaluate_single_prompt(prompt, eval_examples)
            fitness_scores.append(score)
        
        return fitness_scores
    
    async def _evaluate_single_prompt(self, prompt: str, examples: List[Dict]) -> float:
        """Evaluate a single prompt's performance on examples."""
        if not examples:
            return 0.5
        
        correct = 0
        total = len(examples)
        
        for example in examples:
            try:
                # Format prompt with example input
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
                if self._check_answer_correctness(response, expected):
                    correct += 1
                    
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
        
        return correct / total if total > 0 else 0.0
    
    def _check_answer_correctness(self, response: str, expected: str) -> bool:
        """Check if model response contains correct answer."""
        response_clean = re.sub(r'[^\w\s]', '', response.lower().strip())
        expected_clean = re.sub(r'[^\w\s]', '', expected.lower().strip())
        
        # Exact match
        if expected_clean in response_clean:
            return True
        
        # Numerical answers
        response_nums = re.findall(r'\d+(?:\.\d+)?', response)
        expected_nums = re.findall(r'\d+(?:\.\d+)?', expected)
        if response_nums and expected_nums:
            return response_nums[0] == expected_nums[0]
        
        # Multiple choice (A, B, C, D)
        response_choices = re.findall(r'\b[ABCD]\b', response.upper())
        expected_choices = re.findall(r'\b[ABCD]\b', expected.upper())
        if response_choices and expected_choices:
            return response_choices[0] == expected_choices[0]
        
        return False
    
    async def _evolve_population(self, population: List[str], fitness_scores: List[float]) -> List[str]:
        """Create next generation using selection, crossover, and mutation."""
        new_population = []
        
        # Elitism: keep top performers
        elite_count = max(1, self.population_size // 4)
        elite_indices = sorted(range(len(fitness_scores)), 
                             key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(population) >= 2:
                # Crossover
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = await self._crossover(parent1, parent2)
            else:
                # Mutation only
                parent = self._tournament_selection(population, fitness_scores)
                child = await self._mutate_prompt(parent)
            
            new_population.append(child)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[str], fitness_scores: List[float], 
                            tournament_size: int = 3) -> str:
        """Select parent using tournament selection."""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]
    
    async def _crossover(self, parent1: str, parent2: str) -> str:
        """Create offspring by combining two parent prompts using LLM."""
        try:
            crossover_prompt = (
                f"Combine these two prompts into one improved version that captures "
                f"the best aspects of both. Keep the result concise and clear:\n\n"
                f"Prompt 1: {parent1}\n\n"
                f"Prompt 2: {parent2}\n\n"
                f"Combined prompt:"
            )
            
            result = await chat([
                {"role": "system", "content": "You are a prompt optimization expert. Combine prompts effectively while maintaining clarity."},
                {"role": "user", "content": crossover_prompt}
            ], temperature=0.5, max_tokens=200, model=self.model)
            
            offspring = result["text"].strip()
            return offspring if len(offspring) > 10 else parent1
            
        except:
            # Fallback to simple sentence-level crossover
            return self._simple_crossover(parent1, parent2)
    
    def _simple_crossover(self, parent1: str, parent2: str) -> str:
        """Simple crossover when LLM unavailable."""
        sentences1 = [s.strip() for s in parent1.split('.') if s.strip()]
        sentences2 = [s.strip() for s in parent2.split('.') if s.strip()]
        
        if not sentences1:
            return parent2
        if not sentences2:
            return parent1
        
        # Take first half from parent1, second half from parent2
        split_point = len(sentences1) // 2
        combined = sentences1[:split_point] + sentences2[split_point:]
        return '. '.join(combined) + '.'
    
    async def _mutate_prompt(self, prompt: str) -> str:
        """Apply intelligent mutations to a prompt using LLM."""
        if random.random() < 0.5:  # 50% chance of LLM-based mutation
            try:
                mutation_types = [
                    "Make this prompt more specific and detailed:",
                    "Rephrase this prompt for better clarity:", 
                    "Add helpful structure to this prompt:",
                    "Simplify this prompt while keeping the core meaning:",
                    "Make this prompt more engaging:"
                ]
                
                mutation_instruction = random.choice(mutation_types)
                
                result = await chat([
                    {"role": "system", "content": "You are a prompt optimization expert. Apply small improvements to prompts."},
                    {"role": "user", "content": f"{mutation_instruction}\n\n{prompt}"}
                ], temperature=0.6, max_tokens=150, model=self.model)
                
                mutated = result["text"].strip()
                return mutated if len(mutated) > 10 else prompt
                
            except:
                pass
        
        # Fallback to rule-based mutations
        return self._rule_based_mutation(prompt)
    
    def _rule_based_mutation(self, prompt: str) -> str:
        """Apply rule-based mutations when LLM unavailable."""
        mutations = [
            lambda p: p.replace("Please", "Carefully") if "Please" in p else f"Please {p.lower()}",
            lambda p: p.replace("the", "a") if "the" in p else p,
            lambda p: f"Step by step, {p.lower()}" if not p.lower().startswith("step") else p,
            lambda p: p + " Be precise and accurate." if not p.endswith('.') else p.replace('.', '. Be precise and accurate.'),
            lambda p: p.replace("answer", "respond") if "answer" in p else p.replace("solve", "work through") if "solve" in p else p,
            lambda p: f"Think carefully: {p}" if not "think" in p.lower() else p,
            lambda p: p.replace("What", "Determine what") if p.startswith("What") else p
        ]
        
        mutation = random.choice(mutations)
        try:
            return mutation(prompt)
        except:
            return prompt