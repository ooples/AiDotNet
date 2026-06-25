---
title: "PIQABenchmark<T>"
description: "PIQA (Physical Interaction Question Answering) benchmark for physical commonsense reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

PIQA (Physical Interaction Question Answering) benchmark for physical commonsense reasoning.

## For Beginners

PIQA tests whether AI understands how the physical world works
through everyday situations and questions about physical interactions.

**What is PIQA?**
PIQA asks questions about physical commonsense - understanding how objects interact,
what happens when you do certain actions, and basic physics of everyday life.

**Format:**

- Goal: A task you want to accomplish
- Solutions: Two possible ways to do it (one correct, one wrong)
- Question: Which solution actually works?

**Example 1:**
```
Goal: To separate egg whites from the yolk
Solution 1: Crack the egg into a bowl, then use a water bottle to suck up the yolk
Solution 2: Crack the egg and use your hands to throw the white away
Correct: Solution 1
```

**Example 2:**
```
Goal: To remove a stripped screw
Solution 1: Place a rubber band over the screw head for better grip
Solution 2: Pour water on the screw to make it easier to turn
Correct: Solution 1
```

**Example 3:**
```
Goal: Keep your garbage can from smelling bad
Solution 1: Spray perfume in the garbage can every day
Solution 2: Put baking soda at the bottom of the can
Correct: Solution 2
```

**Why it's important:**

- Tests real-world physical understanding
- Can't be solved by language patterns alone
- Requires knowledge of:
- Basic physics (gravity, friction, pressure)
- Material properties (hard, soft, sticky, etc.)
- Cause and effect in physical world
- Practical life skills

**Performance levels:**

- Random guessing: 50%
- Humans: 94.9%
- BERT: 70.9%
- RoBERTa: 79.4%
- GPT-3: 81.0%
- GPT-4: 86.8%
- Claude 3 Opus: 85.2%
- Claude 3.5 Sonnet: 88.0%
- ChatGPT o1: 91.5%

**Categories:**

- Kitchen tasks
- Home repair
- Cleaning
- Arts and crafts
- General household

**Why it's hard for AI:**

- LLMs lack embodied experience
- Can't actually touch or manipulate objects
- Must infer from text descriptions
- Requires implicit physical knowledge

**Research:**

- "PIQA: Reasoning about Physical Commonsense in Natural Language" (Bisk et al., 2020)
- https://arxiv.org/abs/1911.11641
- Dataset: 16,000 questions from WikiHow and other sources
- Part of physical reasoning evaluation suite

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |
| `Description` |  |
| `TotalProblems` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(Func<String,Task<String>>,Nullable<Int32>,CancellationToken)` |  |
| `LoadProblemsAsync(Nullable<Int32>)` |  |

