---
title: "GeneticOptimizer<T>"
description: "Optimizer that uses genetic algorithms to evolve better prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Optimization`

Optimizer that uses genetic algorithms to evolve better prompts.

## For Beginners

Evolves prompts like nature evolves species.

Example:

How it works:

- Start with variations of the initial prompt (population)
- Evaluate how well each performs (fitness)
- Select the best performers
- Combine them (crossover) and introduce random changes (mutation)
- Repeat for many generations

## How It Works

This optimizer mimics natural evolution: prompts are mutated, crossed over,
and selected based on fitness (performance scores).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GeneticOptimizer(Int32,Double,Double,Int32,Nullable<Int32>)` | Initializes a new instance of the GeneticOptimizer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBuildingBlocks(IEnumerable<String>,IEnumerable<String>,IEnumerable<String>)` | Adds custom genetic building blocks for mutation. |
| `OptimizeCore(String,Func<String,>,Int32)` | Optimizes using genetic algorithm. |
| `OptimizeCoreAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Optimizes using genetic algorithm asynchronously. |

