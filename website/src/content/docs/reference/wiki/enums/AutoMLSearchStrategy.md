---
title: "AutoMLSearchStrategy"
description: "Defines the search strategy used to explore AutoML candidate configurations."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the search strategy used to explore AutoML candidate configurations.

## For Beginners

This is how AutoML decides what to try next:

- `RandomSearch` tries random settings (simple and surprisingly strong).
- `BayesianOptimization` tries to learn which settings work best and focus on them.
- `Evolutionary` evolves good settings over time (useful for discrete/conditional knobs).
- `MultiFidelity` uses short runs first and only gives more budget to promising trials.

## How It Works

AutoML can use different strategies to decide which candidate model configurations to try next.
The best choice depends on budget, search-space shape (continuous vs categorical), and how expensive each trial is.

## Fields

| Field | Summary |
|:-----|:--------|
| `BayesianOptimization` | Bayesian optimization (typically Gaussian-process or TPE style). |
| `DARTS` | DARTS (Differentiable Architecture Search) - gradient-based NAS. |
| `Evolutionary` | Evolutionary / genetic search. |
| `GDAS` | GDAS (Gumbel-softmax DARTS) - improved differentiable NAS. |
| `MultiFidelity` | Multi-fidelity search (e.g., HyperBand/ASHA-style scheduling). |
| `NeuralArchitectureSearch` | Neural Architecture Search with automatic algorithm selection. |
| `OnceForAll` | Once-for-All (OFA) Networks - train once, specialize anywhere. |
| `RandomSearch` | Random search baseline. |

