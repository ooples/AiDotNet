---
title: "BayesDAGAlgorithm<T>"
description: "BayesDAG — Bayesian DAG learning with gradient-based posterior inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Bayesian`

BayesDAG — Bayesian DAG learning with gradient-based posterior inference.

## For Beginners

BayesDAG is a modern Bayesian method that efficiently explores
the space of possible causal graphs using gradient-based optimization, providing
principled uncertainty quantification about the causal structure.

## How It Works

BayesDAG uses a DAG-constrained variational framework that maintains a continuous
relaxation of the DAG posterior. It parameterizes edge probabilities via logits Z
and uses the Gumbel-Sigmoid trick for differentiable sampling. The acyclicity
constraint is enforced via augmented Lagrangian on the expected adjacency.

**Algorithm:**

- Initialize edge logits Z[i,j] = 0 (uniform prior on each edge)
- Sample adjacency: A[i,j] ~ Gumbel-Sigmoid(Z[i,j] / tau)
- Compute data likelihood: L = -0.5 * ||X - X*A||^2 / n
- Compute acyclicity: h(A) = tr(e^(A*A)) - d
- Compute ELBO = likelihood - KL(q || prior) - lambda*h(A)
- Update Z via gradient ascent on ELBO
- Anneal temperature tau from soft to hard
- Threshold final sigmoid(Z) to get binary adjacency

Reference: Annadani et al. (2024), "BayesDAG: Gradient-Based Posterior Inference
for Causal Discovery", NeurIPS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

