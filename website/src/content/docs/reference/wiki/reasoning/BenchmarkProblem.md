---
title: "BenchmarkProblem"
description: "Represents a single problem in a benchmark dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks.Models`

Represents a single problem in a benchmark dataset.

## For Beginners

This is one test question with its correct answer.
Think of it like a single problem on a homework assignment or test.

**Example (GSM8K):**
```
Problem: "Janet has 15 apples. She gives 40% to her friend. How many does she have left?"
CorrectAnswer: "9"
```

**Example (HumanEval):**
```
Problem: "Write a function that returns True if a number is prime, False otherwise"
CorrectAnswer: "def is_prime(n): ..." (reference implementation)
```

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` | Category or topic of this problem (e.g., "algebra", "geometry", "sorting"). |
| `CorrectAnswer` | The correct answer or solution. |
| `Difficulty` | Difficulty level (e.g., "easy", "medium", "hard"). |
| `Id` | Unique identifier for this problem. |
| `Metadata` | Additional metadata specific to the benchmark. |
| `Problem` | The problem statement or question. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Gets a summary string of this problem. |

