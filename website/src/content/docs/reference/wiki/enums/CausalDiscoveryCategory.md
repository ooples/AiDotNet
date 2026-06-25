---
title: "CausalDiscoveryCategory"
description: "Categories of causal discovery algorithms based on their methodology."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Categories of causal discovery algorithms based on their methodology.

## For Beginners

Causal discovery algorithms can be grouped by how they work.
Some test statistical independence, some optimize a score, some use continuous math,
and some combine multiple approaches. This enum helps you understand and filter
algorithms by their methodology.

## Fields

| Field | Summary |
|:-----|:--------|
| `Bayesian` | Bayesian methods that maintain a posterior distribution over graph structures. |
| `ConstraintBased` | Constraint-based methods that use conditional independence tests to build the graph skeleton. |
| `ContinuousOptimization` | Continuous optimization methods that formulate DAG learning as a smooth optimization problem. |
| `DeepLearning` | Deep learning methods that use neural networks for structure learning. |
| `Functional` | Functional causal model methods that exploit properties of the noise distribution (e.g., non-Gaussianity). |
| `Hybrid` | Hybrid methods that combine constraint-based and score-based approaches. |
| `InformationTheoretic` | Information-theoretic methods that use entropy and mutual information measures. |
| `ScoreBasedSearch` | Score-based search methods that evaluate candidate DAGs using a scoring function (e.g., BIC, BDeu). |
| `Specialized` | Specialized methods that use unique mathematical formulations (e.g., integer linear programming). |
| `TimeSeries` | Time series causal discovery methods that account for temporal ordering and lagged effects. |

