---
title: "PromptOptimizerBase<T>"
description: "Base class for prompt optimizer implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.PromptEngineering.Optimization`

Base class for prompt optimizer implementations.

## For Beginners

This is the foundation for all prompt optimizers.

It handles:

- Tracking optimization history
- Validation
- Converting prompts to templates
- Providing numeric operations for comparisons

Derived classes implement the optimization algorithm!

## How It Works

This base class provides common functionality for prompt optimizers including history tracking
and validation. Derived classes implement the specific optimization strategy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptOptimizerBase` | Initializes a new instance of the PromptOptimizerBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptimizationHistory` | Gets the optimization history from the most recent optimization run. |
| `Optimize(String,Func<String,>,Int32)` | Optimizes a prompt for a specific task using the provided evaluation function. |
| `OptimizeAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Optimizes a prompt asynchronously for a specific task using the provided evaluation function. |
| `OptimizeCore(String,Func<String,>,Int32)` | Core optimization logic to be implemented by derived classes. |
| `OptimizeCoreAsync(String,Func<String,Task<>>,Int32,CancellationToken)` | Core async optimization logic to be implemented by derived classes. |
| `RecordIteration(Int32,String,)` | Records an iteration in the history. |

## Fields

| Field | Summary |
|:-----|:--------|
| `History` | History of optimization iterations. |
| `NumOps` | Provides numeric operations for the specific type T. |

