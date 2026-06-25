---
title: "BeamSearchOptimizer<T>"
description: "Optimizer that uses beam search to explore multiple promising prompt variations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Optimization`

Optimizer that uses beam search to explore multiple promising prompt variations.

## For Beginners

Explores multiple paths simultaneously.

Example:

How it works:

- Keep track of top N prompts (the "beam")
- Generate variations of all N prompts
- Score all variations
- Keep only the top N again
- Repeat

Benefits:

- More thorough than greedy search
- Less likely to get stuck in local optima
- Faster than exhaustive search

## How It Works

Beam search maintains a fixed number of best candidates at each step,
exploring variations of all candidates before selecting the top performers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BeamSearchOptimizer(Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the BeamSearchOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddVariations(IEnumerable<String>,IEnumerable<String>,IEnumerable<String>)` | Adds custom variations for beam search. |
| `OptimizeCore(String,Func<String,>,Int32)` | Optimizes using beam search. |
| `OptimizeCoreAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Optimizes using beam search asynchronously. |

