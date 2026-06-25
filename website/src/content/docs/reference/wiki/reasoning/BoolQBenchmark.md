---
title: "BoolQBenchmark<T>"
description: "BoolQ benchmark for evaluating yes/no question answering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

BoolQ benchmark for evaluating yes/no question answering.

## For Beginners

BoolQ tests whether AI can answer yes/no questions
about passages of text, requiring reading comprehension.

**What is BoolQ?**
BoolQ (Boolean Questions) contains naturally occurring yes/no questions about
Wikipedia passages. Unlike artificial benchmarks, these are real questions that
people actually asked.

**Format:**

- Passage: A paragraph from Wikipedia
- Question: A yes/no question about the passage
- Answer: True or False

**Example:**
```
Passage: "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars
in Paris, France. It is named after the engineer Gustave Eiffel, whose company
designed and built the tower. Constructed from 1887 to 1889..."

Question: "Is the Eiffel Tower in France?"
Answer: Yes/True

Question: "Was the Eiffel Tower built in the 21st century?"
Answer: No/False
```

**Why it's challenging:**

- Requires careful reading comprehension
- Questions can be tricky or indirect
- Need to distinguish explicit vs. implicit information
- Must avoid making unwarranted inferences
- Real-world questions (not synthetic)

**Performance levels:**

- Random guessing: 50%
- Humans: 89%
- BERT-Large: 77.4%
- RoBERTa: 87.1%
- GPT-3: 76.4%
- GPT-4: 86.9%
- Claude 3 Opus: 87.5%
- Claude 3.5 Sonnet: 91.0%
- ChatGPT o1: 89.5%

**Question types:**

- Factual: Direct facts from the passage
- Inferential: Requires reasoning from passage
- Temporal: About time and dates
- Causal: About cause and effect
- Comparative: About comparisons

**Research:**

- "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions" (Clark et al., 2019)
- https://arxiv.org/abs/1905.10044
- Dataset: 15,942 questions from Google search queries
- Part of SuperGLUE benchmark suite

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

