---
title: "AmortizedCDAlgorithm<T>"
description: "Amortized Causal Discovery — meta-learning approach to causal structure learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

Amortized Causal Discovery — meta-learning approach to causal structure learning.

## For Beginners

Instead of running a slow algorithm each time you have new data,
this approach trains a neural network to recognize causal patterns from data statistics.
The network learns to map correlation/partial-correlation features to edge probabilities.

## How It Works

Amortized Causal Discovery trains a neural network encoder on the input dataset to learn
a mapping from data statistics → edge probabilities. The encoder processes pairwise
sufficient statistics (covariance, partial correlations) through an MLP to produce
edge logits, which are then refined with NOTEARS acyclicity constraint.

**Algorithm:**

- Compute sufficient statistics: covariance matrix and partial correlations
- For each pair (i,j): encode statistics into features via shared MLP
- Edge logits = MLP output, edge probabilities = sigmoid(logits)
- Refine with NOTEARS acyclicity penalty via augmented Lagrangian
- Train encoder end-to-end to minimize reconstruction loss + acyclicity
- Threshold final edge probabilities

Reference: Lowe et al. (2022), "Amortized Causal Discovery: Learning to Infer Causal
Graphs from Time-Series Data", CLeaR.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

