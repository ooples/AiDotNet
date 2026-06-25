---
title: "HumanEvalBenchmark<T>"
description: "HumanEval benchmark for evaluating Python code generation capabilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

HumanEval benchmark for evaluating Python code generation capabilities.

## For Beginners

HumanEval is a benchmark of 164 Python programming problems.
Each problem asks the model to write a function that passes a set of test cases.

**Example problem:**
```
Write a function that returns True if a number is prime, False otherwise.
def is_prime(n: int) -> bool:
# Your code here
```

**Why it's important:**

- Tests code generation abilities
- Requires understanding algorithms
- Tests correctness via unit tests
- Standard benchmark for code models

**Performance levels:**

- GPT-3.5: ~48%
- GPT-4: ~67%
- ChatGPT o1: ~92%
- AlphaCode: ~53%
- CodeGen: ~29%

**Research:**
"Evaluating Large Language Models Trained on Code" (Chen et al., 2021)
https://arxiv.org/abs/2107.03374

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

