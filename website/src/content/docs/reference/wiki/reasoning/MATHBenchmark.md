---
title: "MATHBenchmark<T>"
description: "MATH benchmark for evaluating advanced mathematical reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

MATH benchmark for evaluating advanced mathematical reasoning.

## For Beginners

The MATH dataset contains 12,500 challenging competition mathematics problems
from high school math competitions (AMC, AIME, etc.). These are significantly harder than GSM8K.

**Example problems:**

- "Find the sum of all positive integers n such that sqrt(n^2 + 85) is an integer."
- "A square is inscribed in a circle. What is the ratio of the area of the circle to the square?"
- "Solve the system of equations: x + y + z = 6, xy + xz + yz = 11, xyz = 6"

**Why it's important:**

- Tests advanced mathematical reasoning
- Requires complex multi-step solutions
- Includes algebra, geometry, number theory, calculus
- Benchmark for reasoning capability at competition level

**Performance levels:**

- Human (expert): 90-95%
- GPT-3.5: ~7%
- GPT-4: ~42%
- ChatGPT o1: ~85%
- DeepSeek-R1: ~79.8%
- Minerva (540B): ~50%

**Research:**
"Measuring Mathematical Problem Solving With the MATH Dataset" (Hendrycks et al., 2021)
https://arxiv.org/abs/2103.03874

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

