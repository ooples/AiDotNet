---
title: "CCDrAlgorithm<T>"
description: "CCDr (Concave penalized Coordinate Descent with reparameterization) for DAG learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Functional`

CCDr (Concave penalized Coordinate Descent with reparameterization) for DAG learning.

## For Beginners

CCDr finds causal relationships by fitting a statistical model
where each variable depends on a sparse set of "parent" variables. It uses a clever
penalty (MCP) that encourages most connections to be zero while keeping strong ones
unbiased. The algorithm checks each potential connection one at a time and only keeps
it if removing it would significantly hurt the model fit.

## How It Works

CCDr learns a Bayesian network structure by minimizing a penalized negative log-likelihood
using coordinate descent with a concave penalty (MCP — Minimax Concave Penalty). The MCP
penalty provides near-unbiased estimation of large coefficients while still inducing sparsity.

**Algorithm:**

- Initialize adjacency matrix W = 0
- For each target variable j, cycle through candidate parents i != j
- Update W[i,j] via coordinate descent on the penalized Gaussian log-likelihood
- Apply soft-thresholding with MCP penalty derivative
- Enforce acyclicity: reject updates that would create a cycle
- Repeat until convergence or max iterations

Reference: Aragam and Zhou (2015), "Concave Penalized Estimation of Sparse
Gaussian Bayesian Networks", JMLR.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CCDrAlgorithm(CausalDiscoveryOptions)` | Initializes CCDr with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `HasCycle(Matrix<>,Int32)` | Checks if the adjacency matrix contains a cycle using DFS. |
| `MCPProximal(Double,Double,Double,Double)` | MCP (Minimax Concave Penalty) proximal operator. |

