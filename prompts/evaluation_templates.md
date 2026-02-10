# Evaluation Prompt Templates
# Version: 1.0.0
# Paper Reference: Appendix I
# Last Updated: 2026-01-01

## Target Model System Prompt

```
You are a helpful assistant.
```

## CommonsenseQA Template

Multiple-choice format with letter output constraint.

```
Q: {question}
(A) {choiceA}
(B) {choiceB}
(C) {choiceC}
(D) {choiceD}
(E) {choiceE}
Answer (A/B/C/D/E):
```

### Answer Extraction
Extract a single option letter A–E near the end of the output.
Normalize by stripping punctuation/whitespace.

## HaluEval-QA Template

Evidence-grounded QA format.

```
Knowledge (may be empty):
{knowledge}

Question:
{question}

Answer:
```

## HaluEval-Dialogue Template

Multi-turn conversation format.

```
Conversation:
{dialogue}
Assistant:
```

## HalluLens PreciseWikiQA / LongWiki Template

Wiki-grounded QA format.

```
Context:
{wiki_context}

Question:
{question}

Answer (do not invent facts not supported by the context):
```

## HalluLens NonExistentRefusal Template

Refusal prompt for non-existent entities.

```
User request:
{prompt}

Answer. If the entity or requested information does not exist or cannot be verified, say you do not know rather than guessing:
```

## FActScore Biography Template

Official query form for biography generation.

```
Tell me a bio of {entity}.
```

## Usage Notes

1. All templates should be used with the target model's default system prompt
2. Temperature, top_p, and other decoding parameters should match config/decoding.json
3. Extract answers according to benchmark-specific protocols
4. For CoT prompts (BBH, GSM8K), add "Let's think step by step." to encourage reasoning

