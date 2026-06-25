---
title: "NOTEARSSobolev<T>"
description: "NOTEARS with Sobolev regularization — DAG learning with smoothness constraints."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

NOTEARS with Sobolev regularization — DAG learning with smoothness constraints.

## For Beginners

Regular NOTEARS with neural networks might learn very wiggly functions
that fit noise rather than real causal relationships. The Sobolev penalty encourages smoother
functions, similar to how L2 regularization prevents large weights — but it penalizes the
derivatives (wigglyness) of the learned functions, not just their magnitude.

## How It Works

Extends NOTEARS nonlinear by adding a Sobolev-norm penalty on the functional relationships,
which encourages smooth causal mechanisms. The Sobolev penalty penalizes the squared L2 norm
of the Jacobian (first derivatives) of each MLP, preventing overfitting to noise.

**Algorithm:**

- Initialize per-variable MLPs: Input(d) → Hidden(h, sigmoid) → Output(1)
- Compute L2 reconstruction loss plus Sobolev penalty on Jacobian norms
- Extract adjacency A[i,j] = ||W1[j][:,i]||_2 from input weights
- Apply NOTEARS acyclicity constraint h(A) = tr(e^(A*A)) - d
- Optimize via Adam with augmented Lagrangian for acyclicity
- Threshold the final adjacency matrix

**Sobolev Penalty:** For each MLP f_j, the Sobolev penalty is:
`sum_i (df_j/dx_i)^2` averaged over data samples. This penalizes the sensitivity
of each function to its inputs, encouraging smooth relationships. The Jacobian
df_j/dx_i = sum_k W2[j][k] * sigmoid'(z_k) * W1[j][i,k] is computed analytically.

Reference: Zheng et al. (2020), "Learning Sparse Nonparametric DAGs", AISTATS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NOTEARSSobolev(CausalDiscoveryOptions)` | Initializes NOTEARS Sobolev with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAugLagGradientWithSobolev(Matrix<>,Int32,Int32,Int32,Double,Double)` | Computes augmented Lagrangian gradient with Sobolev penalty. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `ExtractAdjacencyMatrix(Int32)` | Extracts adjacency matrix: A[i,j] = \|\|W1[j][:,i]\|\|_2 |
| `ForwardMLP(Matrix<>,Int32,Int32,Int32,Int32)` | Forward pass: output = W2[j]^T * sigmoid(W1[j]^T * x + b1[j]) + b2[j] Also returns hidden activations and pre-activation values for gradient computation. |

