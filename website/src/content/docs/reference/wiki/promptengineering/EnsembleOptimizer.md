---
title: "EnsembleOptimizer<T>"
description: "Optimizer that combines multiple optimization strategies for better results."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Optimization`

Optimizer that combines multiple optimization strategies for better results.

## For Beginners

Like getting multiple opinions and combining them.

Example:

How it works:

- Run multiple optimizers independently
- Collect their best results
- Pick the overall best (or combine them)

Benefits:

- More robust than single strategy
- Different strategies find different optima
- Better coverage of solution space

## How It Works

This optimizer runs multiple optimization strategies and combines their results,
using voting, averaging, or selection to determine the best prompt.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnsembleOptimizer(EnsembleOptimizer<>.EnsembleStrategy,IPromptOptimizer<>[])` | Initializes a new instance of the EnsembleOptimizer class with strategy and optimizers. |
| `EnsembleOptimizer(IPromptOptimizer<>[])` | Initializes a new instance of the EnsembleOptimizer class with specified optimizers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddOptimizer(IPromptOptimizer<>)` | Adds an optimizer to the ensemble. |
| `CreateAggressive` | Creates an aggressive ensemble with more optimizers. |
| `CreateDefault` | Creates a default ensemble with common optimization strategies. |
| `OptimizeCore(String,Func<String,>,Int32)` | Optimizes using ensemble strategies. |
| `OptimizeCoreAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Optimizes using ensemble strategies asynchronously. |

