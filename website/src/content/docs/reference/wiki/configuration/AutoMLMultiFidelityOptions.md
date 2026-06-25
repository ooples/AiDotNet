---
title: "AutoMLMultiFidelityOptions"
description: "Configuration options for multi-fidelity/ASHA AutoML search."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for multi-fidelity/ASHA AutoML search.

## For Beginners

Instead of fully training every trial (slow), multi-fidelity does:

- Train many candidates on a small amount of data.
- Keep only the best candidates.
- Train those candidates on more data.
- Repeat until you reach full training.

This usually finds strong models faster than running full training for every attempt.

## How It Works

Multi-fidelity search tries many configurations quickly with a smaller "budget" (for example, a subset of the
training data) and then promotes only the most promising trials to larger budgets.

ASHA (Asynchronous Successive Halving Algorithm) extends this with parallel trial execution and
early stopping of underperforming trials, providing 5-10x speedup over grid/random search.

## Properties

| Property | Summary |
|:-----|:--------|
| `EarlyStoppingMinDelta` | Gets or sets the minimum improvement threshold for early stopping. |
| `EarlyStoppingPatience` | Gets or sets the early stopping patience for individual trials within a rung. |
| `EnableAsyncExecution` | Gets or sets whether to enable ASHA-style async parallel trial execution. |
| `EnableHyperBandBrackets` | Gets or sets whether to use aggressive bracket halving (HyperBand-style). |
| `GracePeriod` | Gets or sets the grace period (minimum checkpoints) before early stopping can trigger. |
| `HyperBandBrackets` | Gets or sets the number of HyperBand brackets to use when `EnableHyperBandBrackets` is true. |
| `MaxParallelism` | Gets or sets the maximum number of trials to run in parallel at each fidelity rung. |
| `ReductionFactor` | Gets or sets the reduction factor used when promoting trials between fidelity levels. |
| `TrainingFractions` | Gets or sets the ordered list of training-data fractions to use as fidelity levels. |

