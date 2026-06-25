---
title: "IterativeMCMCAlgorithm<T>"
description: "Iterative MCMC — iteratively refined MCMC for Bayesian network structure learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Bayesian`

Iterative MCMC — iteratively refined MCMC for Bayesian network structure learning.

## For Beginners

Standard MCMC can get "stuck" in local optima. Iterative MCMC
uses clever tricks to escape these traps and explore more of the solution space,
leading to better posterior estimates.

## How It Works

Iterative MCMC improves MCMC mixing by alternating between different proposal mechanisms:
edge additions, deletions, and reversals. It uses multiple restarts with the best DAG
from each restart used to seed the next, progressively refining the search.

**Algorithm:**

- Initialize DAG from empty graph or previous restart's best
- Propose: randomly add, delete, or reverse an edge
- Check acyclicity of proposed DAG
- Compute BIC score difference
- Accept/reject via Metropolis-Hastings
- After burn-in, accumulate edge posterior probabilities
- Restart with best DAG found so far as seed

Reference: Kuipers et al. (2017), "Efficient Structure Learning and Sampling of
Bayesian Networks", arXiv.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

