---
title: "IBenchmark<T>"
description: "Defines the contract for reasoning benchmarks that evaluate model performance."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for reasoning benchmarks that evaluate model performance.

## For Beginners

A benchmark is like a standardized test for AI reasoning systems.
Just like students take SAT or ACT tests to measure their abilities, AI systems are evaluated
on benchmarks to measure their reasoning capabilities.

**Common benchmarks:**

- **GSM8K**: Grade school math problems (8,000 questions)
- **MATH**: Competition-level mathematics
- **HumanEval**: Code generation tasks
- **MMLU**: Multiple choice questions across many subjects
- **ARC-AGI**: Abstract reasoning puzzles

**Why benchmarks matter:**

- Objective measurement of performance
- Compare different approaches
- Track improvements over time
- Identify strengths and weaknesses

**Example:**
```cs
var benchmark = new GSM8KBenchmark<double>();
var results = await benchmark.EvaluateAsync(reasoner, sampleSize: 100);
// Result is available in the returned value // "Accuracy: 87.5%"
```

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` | Gets the name of this benchmark. |
| `Description` | Gets a description of what this benchmark measures. |
| `TotalProblems` | Gets the total number of problems in this benchmark. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(Func<String,Task<String>>,Nullable<Int32>,CancellationToken)` | Evaluates a reasoning strategy on this benchmark. |
| `LoadProblemsAsync(Nullable<Int32>)` | Loads benchmark problems (for inspection or custom evaluation). |

