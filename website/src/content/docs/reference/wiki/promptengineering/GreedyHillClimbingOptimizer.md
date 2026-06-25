---
title: "GreedyHillClimbingOptimizer<T>"
description: "Simple greedy optimizer that always moves toward better solutions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Optimization`

Simple greedy optimizer that always moves toward better solutions.

## For Beginners

Always takes the best step available.

Example:

How it works:

- Try a variation of current prompt
- If it's better, use it as the new current
- If it's worse, try another variation
- Repeat until no improvements found

Benefits:

- Simple and fast
- Guaranteed to improve (or stay same)
- Good for fine-tuning near a good solution

Limitations:

- Can get stuck at local optima
- May miss better solutions further away

## How It Works

Hill climbing is a simple optimization strategy that always accepts improvements
and never accepts worse solutions. Fast but can get stuck in local optima.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GreedyHillClimbingOptimizer(Int32,Nullable<Int32>)` | Initializes a new instance of the GreedyHillClimbingOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `OptimizeCore(String,Func<String,>,Int32)` | Optimizes using greedy hill climbing. |
| `OptimizeCoreAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Optimizes using greedy hill climbing asynchronously. |

