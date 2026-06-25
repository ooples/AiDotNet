---
title: "NOTEARSNonlinear<T>"
description: "NOTEARS Nonlinear — continuous optimization for DAG structure learning with nonlinear (MLP) relationships."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

NOTEARS Nonlinear — continuous optimization for DAG structure learning with nonlinear (MLP) relationships.

## For Beginners

The linear version assumes each variable is a weighted sum of its parents.
The nonlinear version uses small neural networks instead, so it can capture curved or complex
relationships between variables. For example, if income depends on age in a U-shaped curve,
the nonlinear version can learn that while the linear version cannot.

## How It Works

Extends the NOTEARS framework to handle nonlinear causal relationships by replacing
the linear model X = XW + noise with a nonlinear model X_j = f_j(Pa(X_j)) + noise,
where each f_j is parameterized by a small MLP (multi-layer perceptron).

**Key Differences from Linear NOTEARS:**

- Each variable's structural equation is modeled by a 2-layer MLP
- The adjacency matrix is derived from the input-layer weights: A[i,j] = ||W1_j[:,i]||_2
- Acyclicity constraint: h(theta) = tr(e^(A∘A)) - d = 0, using the derived adjacency
- Optimization over MLP parameters theta, not over a weight matrix W directly

**Architecture per variable:** Input (d neurons) → Hidden (h neurons, sigmoid) → Output (1 neuron).
The "adjacency matrix" is extracted from the input weights: if the weights from variable i to
variable j's MLP are small, there's no edge i → j.

Reference: Zheng et al. (2020), "Learning Sparse Nonparametric DAGs", AISTATS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NOTEARSNonlinear(CausalDiscoveryOptions)` | Initializes NOTEARS Nonlinear with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `ExtractAdjacencyMatrix(Int32)` | Extract adjacency matrix from MLP weights: A[i,j] = \|\|W1[j][:,i]\|\|_2 |
| `ForwardMLP(Matrix<>,Int32,Int32,Int32)` | Forward pass for variable j: output = W2[j]^T * sigmoid(W1[j]^T * x + b1[j]) + b2[j] |
| `GetLastRunInfo` | Gets the iteration count and convergence info from the last run. |

