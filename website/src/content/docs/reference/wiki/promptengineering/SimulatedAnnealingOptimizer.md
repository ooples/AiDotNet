---
title: "SimulatedAnnealingOptimizer<T>"
description: "Optimizer that uses simulated annealing to escape local optima."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Optimization`

Optimizer that uses simulated annealing to escape local optima.

## For Beginners

Like cooling metal - starts flexible, becomes rigid.

Example:

How it works:

- High temperature: Accept many variations, even worse ones
- Cooling down: Gradually become more selective
- Low temperature: Only accept improvements (like greedy search)

Benefits:

- Escapes local optima by exploring broadly
- Proven effective for combinatorial optimization
- Balances exploration and exploitation

## How It Works

Simulated annealing occasionally accepts worse solutions early in the search
(when "temperature" is high), allowing exploration of the solution space.
As temperature decreases, it becomes more greedy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimulatedAnnealingOptimizer(Double,Double,Double,Nullable<Int32>)` | Initializes a new instance of the SimulatedAnnealingOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `OptimizeCore(String,Func<String,>,Int32)` | Optimizes using simulated annealing. |
| `OptimizeCoreAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Optimizes using simulated annealing asynchronously. |

