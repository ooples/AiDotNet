---
title: "GraNDAGAlgorithm<T>"
description: "GraN-DAG — Gradient-based Neural DAG Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

GraN-DAG — Gradient-based Neural DAG Learning.

## For Beginners

GraN-DAG trains a separate neural network for each variable to
predict it from the others. The "importance" of each input connection tells us the
causal strength, while a mathematical constraint ensures no circular causation.

## How It Works

GraN-DAG parameterizes each structural equation f_j as a neural network with sigmoid
activations. The weighted adjacency matrix A[i,j] = ||W1_j[:,i]||_2 is derived from
the first-layer input weights. Path-specific connectivity through the MLP gives a
refined adjacency measure. The NOTEARS acyclicity constraint h(A) = tr(e^(A*A)) - d
is enforced via augmented Lagrangian.

Reference: Lachapelle et al. (2020), "Gradient-Based Neural DAG Learning", ICLR.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

