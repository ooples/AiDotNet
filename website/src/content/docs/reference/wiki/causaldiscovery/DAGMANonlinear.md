---
title: "DAGMANonlinear<T>"
description: "DAGMA Nonlinear — DAG learning via M-matrices and log-determinant with MLP structural equations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

DAGMA Nonlinear — DAG learning via M-matrices and log-determinant with MLP structural equations.

## For Beginners

This is like DAGMA Linear but can discover non-linear causal relationships
using small neural networks. While DAGMA Linear assumes Y = a*X + b, this version can learn
curved relationships like Y = sin(X) or Y = X^2.

## How It Works

Extends DAGMA to nonlinear settings by parameterizing each structural equation with a small MLP.
Uses the same log-determinant acyclicity constraint as DAGMA Linear but applied to the adjacency
matrix extracted from MLP input-layer weights.

**Constraint:** h(theta, s) = -log det(sI - A∘A) + d*log(s), where A[i,j] = ||W1_j[:,i]||_2

Reference: Bello et al. (2022), "DAGMA: Learning DAGs via M-matrices and a Log-Determinant
Acyclicity Characterization", NeurIPS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `GetLastRunInfo` | Gets the iteration count and convergence info from the last run. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_seed` | Initializes DAGMA Nonlinear with optional configuration. |

