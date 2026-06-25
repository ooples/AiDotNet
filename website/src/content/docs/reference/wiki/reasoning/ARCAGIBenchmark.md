---
title: "ARCAGIBenchmark<T>"
description: "ARC-AGI (Abstract Reasoning Corpus - Artificial General Intelligence) benchmark."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

ARC-AGI (Abstract Reasoning Corpus - Artificial General Intelligence) benchmark.

## For Beginners

ARC-AGI is considered one of the hardest AI benchmarks.
It tests abstract reasoning and pattern recognition using visual grid puzzles.

**What is ARC?**
Created by François Chollet (creator of Keras), ARC tests whether AI can think
abstractly like humans. Each task shows example input/output grids, and the AI
must figure out the transformation rule.

**Example task:**
```
Training examples:
Input: [1,1,0] Output: [2,2,0]
[1,1,0] [2,2,0]
[0,0,0] [0,0,0]

Input: [0,1,1] Output: [0,2,2]
[0,1,1] [0,2,2]
[0,0,0] [0,0,0]

Test (what's the output?):
Input: [1,1,1] Output: ???
[0,0,0]
[0,0,0]

Rule: Replace all 1s with 2s
```

**Why it's hard:**

- Requires understanding abstract concepts
- Few-shot learning (only 2-3 examples)
- Novel tasks never seen before
- Can't be solved by memorization
- Tests core intelligence, not just pattern matching

**Performance levels:**

- Human performance: ~85%
- GPT-4: ~0-5% (very poor)
- GPT-4o: ~10%
- Claude 3.5 Sonnet: ~15-20%
- ChatGPT o1: ~21%
- Specialized systems: ~20-30%
- **Current SOTA**: ~55% (MindsAI ARC Prize 2024)

**Why LLMs struggle:**

1. Pattern recognition ≠ abstract reasoning
2. Can't generalize from few examples
3. No spatial/visual reasoning built-in
4. Trained on language, not logic puzzles

**Recent progress:**

- ARC Prize (2024): $1M prize for solving ARC
- Test-time compute scaling helps (o1, o3)
- Hybrid neuro-symbolic approaches show promise

**Research:**

- "On the Measure of Intelligence" (Chollet, 2019) - Original ARC paper
- "The ARC Prize" (2024) - Competition for AGI progress
- ARC-AGI is viewed as a benchmark for measuring progress toward AGI

**Categories of tasks:**

- Object counting and manipulation
- Symmetry and patterns
- Color transformations
- Spatial reasoning
- Logical rules

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

