---
title: "CGNNAlgorithm<T>"
description: "CGNN — Causal Generative Neural Networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

CGNN — Causal Generative Neural Networks.

## For Beginners

CGNN tests different causal graph candidates by asking "If this
graph were correct, could a neural network generate data that looks like the real data?"
The graph that produces the most realistic synthetic data is chosen as the answer.

## How It Works

CGNN generates data according to a causal model parameterized by neural networks.
For each candidate edge (i→j), a generative model f_j(parents(j), noise) is trained.
The model quality is measured by Maximum Mean Discrepancy (MMD) between generated
and observed data. Edges are scored pairwise: the direction with lower MMD indicates
the causal direction.

**Algorithm:**

- Initialize from correlation-based skeleton
- For each pair (i,j) with non-zero correlation:
- Train MLP f: x_i + noise → x_j, compute MMD(generated_j, real_j)
- Train MLP g: x_j + noise → x_i, compute MMD(generated_i, real_i)
- If MMD(f) < MMD(g), edge is i→j; otherwise j→i
- Compute OLS weights for the oriented edges

Reference: Goudet et al. (2018), "Learning Functional Causal Models with Generative
Neural Networks", Explainable and Interpretable Models in CV and ML.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `ForwardMLP(,,Matrix<>,Matrix<>,Int32,Vector<>,Vector<>)` | Forward pass through the 2-input MLP: sigmoid hidden layer, linear output. |
| `TrainAndComputeMMD(Matrix<>,Int32,Int32,Int32,Int32,Random)` | Trains a small MLP to predict target from source+noise and returns the distribution distance (mean + variance discrepancy) between predicted and actual target values. |

