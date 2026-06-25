---
title: "GSM8KBenchmark<T>"
description: "Grade School Math 8K (GSM8K) benchmark for evaluating mathematical reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

Grade School Math 8K (GSM8K) benchmark for evaluating mathematical reasoning.

## For Beginners

GSM8K is a dataset of 8,500 grade school math word problems.
These are the kinds of problems you'd see in elementary school math:

**Example problems:**

- "Janet has 15 apples. She gives 40% to her friend. How many does she have left?"
- "A train travels 60 mph for 2.5 hours. How far does it go?"
- "If 3 pizzas cost $45, how much does 1 pizza cost?"

**Why it's important:**

- Tests basic mathematical reasoning
- Requires understanding word problems
- Needs step-by-step calculation
- Benchmark for many reasoning models

**Performance levels:**

- Human performance: ~90-95%
- GPT-3.5: ~57%
- GPT-4: ~92%
- ChatGPT o1: ~95%
- DeepSeek-R1: ~97%

**Research:**
"Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
https://arxiv.org/abs/2110.14168

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GSM8KBenchmark` | Initializes a new instance of the `GSM8KBenchmark` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |
| `Description` |  |
| `TotalProblems` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CompareAnswers(String,String)` | Compares two numerical answers with tolerance. |
| `EvaluateAsync(Func<String,Task<String>>,Nullable<Int32>,CancellationToken)` |  |
| `ExtractNumericalAnswer(String)` | Extracts numerical answer from text. |
| `GenerateSampleProblems` | Generates sample GSM8K-style problems for demonstration. |
| `LoadProblemsAsync(Nullable<Int32>)` |  |

