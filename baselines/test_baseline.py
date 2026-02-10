#!/usr/bin/env python3
"""
Test script for individual baselines
"""

import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from baselines.evoprompt import EvoPromptBaseline
from baselines.opro import OPROBaseline
from baselines.promptwizard import PromptWizardBaseline
from baselines.apo import APOBaseline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_baseline(baseline_name: str):
    """Test a specific baseline"""
    logger.info(f"Testing {baseline_name} baseline...")
    
    # Create test examples
    test_examples = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"}
    ]
    
    initial_prompt = "Answer the following question:"
    
    try:
        if baseline_name == "evoprompt":
            baseline = EvoPromptBaseline(generations=2, population_size=4)
        elif baseline_name == "opro":
            baseline = OPROBaseline(max_iterations=3, num_candidates=3)
        elif baseline_name == "promptwizard":
            baseline = PromptWizardBaseline(num_rounds=2, num_candidates=3)
        elif baseline_name == "apo":
            baseline = APOBaseline(rounds=2, beam_size=3, minibatch_size=4)
        else:
            logger.error(f"Unknown baseline: {baseline_name}")
            return False
            
        # Test optimization (use sync method to avoid asyncio.run() issues)
        optimized_prompt = baseline.optimize_prompt(initial_prompt, test_examples)
        
        logger.info(f"{baseline_name} optimization completed successfully")
        logger.info(f"Initial prompt: {initial_prompt}")
        logger.info(f"Optimized prompt: {optimized_prompt[:100]}...")
        return True
        
    except Exception as e:
        logger.error(f"{baseline_name} test failed: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_baseline.py <baseline_name>")
        print("Available baselines: evoprompt, opro, promptwizard, apo")
        sys.exit(1)
        
    baseline_name = sys.argv[1].lower()
    
    try:
        success = test_baseline(baseline_name)
        
        if success:
            logger.info(f"✓ {baseline_name} baseline test passed")
        else:
            logger.error(f"✗ {baseline_name} baseline test failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()