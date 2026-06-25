---
title: "BayesianCausalBase<T>"
description: "Base class for Bayesian causal discovery algorithms."
section: "API Reference"
---

`Base Classes` · `AiDotNet.CausalDiscovery.Bayesian`

Base class for Bayesian causal discovery algorithms.

## For Beginners

Instead of returning a single "best guess" graph, Bayesian methods
give probabilities for each possible connection. This tells you not just "X probably
causes Y" but also "we're 90% confident about this." The trade-off is higher computation.

## How It Works

Bayesian methods maintain a posterior distribution over possible DAG structures given
the data. They can represent uncertainty about the causal structure and are naturally
suited for model averaging. Methods include MCMC sampling over graphs, variational
inference, and gradient-based approaches.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `NumSamples` | Number of MCMC samples or variational iterations. |
| `Seed` | Random seed for reproducibility. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyBayesianOptions(CausalDiscoveryOptions)` | Applies Bayesian-specific options. |

