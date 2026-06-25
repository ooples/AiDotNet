---
title: "BenchmarkResult<T>"
description: "Results from evaluating a reasoning system on a benchmark."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks.Models`

Results from evaluating a reasoning system on a benchmark.

## For Beginners

This is like a report card for your reasoning system's performance
on a standardized test.

**Key metrics:**

- **Accuracy**: Percentage of problems answered correctly (most important)
- **Total Evaluated**: How many problems were tested
- **Correct Count**: How many were answered correctly
- **Average Confidence**: How confident the system was on average

**Example:**
```
Benchmark: GSM8K (Grade School Math)
Problems Evaluated: 100
Correct: 87
Accuracy: 87.0%
Average Confidence: 0.92
Average Time: 3.2 seconds per problem
```

This would indicate the system got 87 out of 100 math problems correct, with high confidence.

## Properties

| Property | Summary |
|:-----|:--------|
| `Accuracy` | Overall accuracy (correct / total) as a value between 0.0 and 1.0. |
| `AccuracyByCategory` | Breakdown of accuracy by category (if applicable). |
| `AverageConfidence` | Average confidence across all evaluated problems. |
| `AverageTimePerProblem` | Average time per problem. |
| `BenchmarkName` | Name of the benchmark that was evaluated. |
| `ConfidenceScores` | Confidence scores for each evaluated problem (as a Vector). |
| `CorrectCount` | Number of problems answered correctly. |
| `Metrics` | Additional benchmark-specific metrics. |
| `ProblemResults` | Detailed results for each evaluated problem. |
| `TotalDuration` | Total time spent evaluating all problems. |
| `TotalEvaluated` | Total number of problems evaluated. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a summary string of the benchmark results. |
| `ToString` | Returns a summary string. |

