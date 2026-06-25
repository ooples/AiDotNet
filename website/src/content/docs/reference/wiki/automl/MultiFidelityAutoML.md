---
title: "MultiFidelityAutoML<T, TInput, TOutput>"
description: "Built-in AutoML strategy that uses multi-fidelity (successive halving) and ASHA scheduling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML`

Built-in AutoML strategy that uses multi-fidelity (successive halving) and ASHA scheduling.

## For Beginners

This is a "try cheap first, then spend more on the best" strategy:

- Try many models quickly (small subset).
- Keep the best few.
- Re-train those on more data.
- Repeat until full training data.

## How It Works

This strategy evaluates many candidate configurations on a reduced training budget first (for example, a smaller
subset of rows), then promotes only the most promising trials to higher budgets.

**ASHA (Asynchronous Successive Halving Algorithm)** extends this with:

- Parallel trial execution at each fidelity rung.
- Per-trial early stopping of underperforming configurations.
- Grace periods to allow trials to "warm up" before stopping.

Enable ASHA via `EnableAsyncExecution`.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExecuteSingleTrialAsync(Dictionary<String,Object>,,,,,Double,Int32,Int32[],CancellationToken)` | Executes a single trial and returns the score and success status. |
| `ExecuteTrialsInParallelAsync(List<Dictionary<String,Object>>,,,,,Double,Int32,Int32[],List<ValueTuple<Dictionary<String,Object>,Double,Boolean>>,Object,Int32,DateTime,IFullModel<,,>,Double,CancellationToken)` | Executes trials in parallel using ASHA-style async execution. |

