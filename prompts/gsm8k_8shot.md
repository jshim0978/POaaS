# GSM8K 8-Shot Chain-of-Thought Prompt Template
# Version: 1.0.0
# Paper Reference: Appendix I
# Last Updated: 2026-01-01

## Overview

GSM8K uses an 8-shot Chain-of-Thought prompt for math word problems.
Each example demonstrates step-by-step mathematical reasoning.

## System Context

As an expert problem solver, solve step by step the following mathematical questions.

## 8-Shot Demonstrations

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step.
There are 15 trees originally.
After planting, there are 21 trees.
So the workers planted 21 - 15 = 6 trees.
The final answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step.
There are originally 3 cars.
2 more cars arrive.
So there are 3 + 2 = 5 cars in the parking lot.
The final answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step.
Originally, Leah had 32 chocolates.
Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they have 74 - 35 = 39 pieces.
The final answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step.
Jason started with 20 lollipops.
Now he has 12 lollipops.
So he gave away 20 - 12 = 8 lollipops.
The final answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Let's think step by step.
Shawn started with 5 toys.
He got 2 toys from his mom.
He got 2 toys from his dad.
So he got 2 + 2 = 4 toys total.
Now he has 5 + 4 = 9 toys.
The final answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?
A: Let's think step by step.
There were originally 9 computers.
5 computers were installed each day from Monday to Thursday.
That's 4 days total.
So 5 * 4 = 20 computers were installed.
Now there are 9 + 20 = 29 computers.
The final answer is 29.

Q: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?
A: Let's think step by step.
Michael started with 58 golf balls.
On Tuesday, he lost 23.
So after Tuesday he had 58 - 23 = 35 golf balls.
On Wednesday, he lost 2 more.
So he had 35 - 2 = 33 golf balls.
The final answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's think step by step.
Olivia had $23.
She bought 5 bagels at $3 each.
So she spent 5 * 3 = $15.
She has 23 - 15 = $8 left.
The final answer is 8.

## Template Format

```
As an expert problem solver, solve step by step the following mathematical questions.

{GSM8K_8shot_cot_demos}

Q: {question}
A: Let's think step by step.
```

## Answer Extraction

The final answer should be extracted from "The final answer is X" format.
Normalize to numeric value (remove $ signs, commas, units).

