# BBH 3-Shot Chain-of-Thought Prompt Templates
# Version: 1.0.0
# Paper Reference: Appendix I
# Last Updated: 2026-01-01

## Overview

Big Bench Hard (BBH) uses task-specific 3-shot Chain-of-Thought prompts.
Each task has its own prefix with examples demonstrating step-by-step reasoning.

## Template Format

```
{BBH_3shot_cot_prefix}

Q: {question}
A: Let's think step by step.
```

## Task-Specific Prefixes

### Boolean Expressions

Evaluate the result of a random Boolean expression.

Q: not ( ( not not True ) ) is
A: Let's think step by step.
not not True = True
not ( True ) = False
So the answer is False.

Q: True and False and not True and True is
A: Let's think step by step.
not True = False
True and False = False
False and False = False
False and True = False
So the answer is False.

Q: not not ( not ( False ) ) is
A: Let's think step by step.
not ( False ) = True
not ( True ) = False
not False = True
So the answer is True.

### Date Understanding

Infer the date from context.

Q: Yesterday was April 30, 2021. What is the date one week from today in MM/DD/YYYY?
A: Let's think step by step.
Yesterday was April 30, 2021, so today is May 1, 2021.
One week from today would be May 8, 2021.
The answer is 05/08/2021.

Q: The concert was scheduled for 06/01/1943, but was delayed by one day. What is the date in MM/DD/YYYY?
A: Let's think step by step.
The original date was 06/01/1943.
Delayed by one day means 06/02/1943.
The answer is 06/02/1943.

Q: It is 4/19/1969 today. What is the date one week ago from today in MM/DD/YYYY?
A: Let's think step by step.
Today is 4/19/1969.
One week ago would be 4/12/1969.
The answer is 04/12/1969.

### Causal Judgement

Answer questions about causal attribution.

Q: How would a typical person answer each of the following questions about causation?
Frank T., had an ongoing dispute with his neighbor over Frank's backyard overhanging shed. One day, Frank's neighbor asked him to remove the shed. Frank got angry and set fire to his own house. Did Frank cause the fire?
A: Let's think step by step.
Frank directly performed the action of setting fire to his house.
Even though he was provoked by his neighbor, he made the choice to start the fire.
A typical person would say Frank caused the fire because he lit it.
The answer is Yes.

Q: How would a typical person answer each of the following questions about causation?
A machine is designed to detect defective products and sort them out. The machine fails one time in a million. One day, a defective product got through. Did the machine cause the defective product to get through?
A: Let's think step by step.
The machine is designed to prevent defective products from getting through.
It failed to catch this one, but the defect existed before reaching the machine.
The machine's failure allowed it through, but didn't cause the defect.
A typical person would say the machine did not cause it.
The answer is No.

Q: How would a typical person answer each of the following questions about causation?
Billy and Suzy both throw rocks at a bottle. Suzy's rock gets there first and shatters the bottle. Billy's rock arrives after. Did Billy cause the bottle to shatter?
A: Let's think step by step.
Suzy's rock hit the bottle first and broke it.
Billy's rock arrived after the bottle was already broken.
Billy's throw did not actually break the bottle.
A typical person would say Billy did not cause the bottle to shatter.
The answer is No.

## Usage Notes

- Use the task-appropriate prefix based on the BBH subtask
- Always end with "Let's think step by step."
- The model should generate reasoning before the final answer
- Extract the final answer after "The answer is" or "So the answer is"

