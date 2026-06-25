---
title: "NOTEARSLowRank<T>"
description: "NOTEARS Low-Rank — DAG learning with low-rank parameterization for scalability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

NOTEARS Low-Rank — DAG learning with low-rank parameterization for scalability.

## For Beginners

When you have many variables (say hundreds), the standard NOTEARS
weight matrix becomes very large. The low-rank trick says "most causal graphs are relatively
simple" and represents the matrix using fewer numbers, making the optimization much faster.

## How It Works

Parameterizes the weighted adjacency matrix W as a product of low-rank factors W = A * B^T,
where A and B are d x r matrices with r much less than d. This reduces the number of
parameters from O(d^2) to O(dr) and enables scalability to graphs with many variables.
The NOTEARS acyclicity constraint h(W) = tr(e^(W * W)) - d is applied to the reconstructed W.

**Algorithm:**

- Choose rank r = min(d, 10)
- Initialize A, B as small random matrices of size d x r
- Reconstruct W = A * B^T (with diagonal zeroed)
- Compute L2 loss and NOTEARS acyclicity constraint on W
- Compute gradients via chain rule: dL/dA = dL/dW * B, dL/dB = dL/dW^T * A
- Update A, B via gradient descent with augmented Lagrangian for acyclicity
- Threshold the final W = A * B^T

Reference: Fang et al. (2023), "On Low-Rank Directed Acyclic Graphs and Causal
Structure Learning", IEEE Transactions on Signal Processing.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NOTEARSLowRank(CausalDiscoveryOptions)` | Initializes NOTEARS Low-Rank with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `FlattenABToVector(Matrix<>,Matrix<>,Int32,Int32)` | Flattens low-rank factors [A; B] into a single Vector for the optimizer. |
| `ReconstructW(Matrix<>,Matrix<>,Int32,Int32)` | Reconstructs W = A * B^T with diagonal zeroed. |
| `UnflattenVectorToAB(Vector<>,Matrix<>,Matrix<>,Int32,Int32)` | Unflattens a parameter Vector back into A, B matrices. |

