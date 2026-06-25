---
title: "MetaPACOHOptions<T, TInput, TOutput>"
description: "Configuration options for Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning with per-group prior variances."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Meta-PACOH: Hierarchical PAC-Bayesian Meta-Learning
with per-group prior variances.

## How It Works

Meta-PACOH extends PACOH by introducing a hierarchical Bayesian structure where
different parameter groups (e.g., layers) have independently learned prior variances.
This enables the algorithm to express that some parameter groups should be more
tightly constrained to the meta-learned prior while others can vary freely.

The algorithm meta-learns both a shared prior mean μ_P and per-group prior
log-variances {log(σ²_g)} to minimize a hierarchical PAC-Bayesian bound.

## Properties

| Property | Summary |
|:-----|:--------|
| `Delta` | Confidence parameter δ for the PAC-Bayesian bound. |
| `HyperPriorLogVar` | Hyper-prior log-variance controlling how much per-group variances can deviate. |
| `InitialLogVariance` | Initial log-variance for all prior groups. |
| `KLCoefficient` | KL divergence coefficient for the hierarchical PAC-Bayesian bound. |
| `NumPriorGroups` | Number of parameter groups with independent prior variances. |

