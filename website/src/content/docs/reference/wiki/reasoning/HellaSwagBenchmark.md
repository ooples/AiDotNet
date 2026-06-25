---
title: "HellaSwagBenchmark<T>"
description: "HellaSwag benchmark for evaluating commonsense natural language inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

HellaSwag benchmark for evaluating commonsense natural language inference.

## For Beginners

HellaSwag tests whether AI can predict what happens next
in everyday situations using common sense.

**What is HellaSwag?**
HellaSwag (Harder Endings, Longer contexts, and Low-shot Activities for Situations With
Adversarial Generations) presents a context and asks the model to choose the most
plausible continuation from 4 options.

**Example:**
```
Context: "A woman is sitting at a piano. She"
Options:
A) sits on a bench and plays the piano
B) starts to play the piano with her feet
C) pulls out a sandwich from the piano
D) transforms into a dolphin

Answer: A (most plausible)
```

**Why it's called "HellaSwag"?**

- "Hella" = very (slang)
- Designed to be harder than previous benchmarks (SWAG)
- Uses adversarial generation to create tricky wrong answers

**Categories:**

- ActivityNet: Video descriptions (activities)
- WikiHow: Instructional text (how-to guides)

**Adversarial wrong answers:**
The wrong options are generated to be:

1. Grammatically plausible
2. Semantically similar
3. But factually/logically incorrect

This makes random guessing ineffective and requires actual understanding.

**Performance levels:**

- Random guessing: 25%
- Humans: 95.6%
- BERT-Large: 47.9%
- GPT-3: 78.9%
- GPT-4: 95.3%
- Claude 3 Opus: 88.0%
- Claude 3.5 Sonnet: 89.0%
- ChatGPT o1: 94.2%

**Why it's hard for models:**

- Requires real-world common sense
- Can't be solved by pattern matching alone
- Adversarial wrong answers look plausible
- Needs understanding of cause and effect

**Research:**

- "HellaSwag: Can a Machine Really Finish Your Sentence?" (Zellers et al., 2019)
- https://arxiv.org/abs/1905.07830
- Dataset: 70,000 questions from ActivityNet and WikiHow

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

