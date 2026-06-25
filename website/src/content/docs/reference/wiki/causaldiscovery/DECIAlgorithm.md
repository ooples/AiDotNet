---
title: "DECIAlgorithm<T>"
description: "DECI — Deep End-to-end Causal Inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

DECI — Deep End-to-end Causal Inference.

## For Beginners

DECI simultaneously learns "which variables cause which" and
"how they cause each other." It's particularly good at handling complex, non-standard
relationships and can also estimate intervention effects.

## How It Works

DECI is a flow-based variational inference method that jointly learns the causal graph
and the functional relationships between variables. It uses normalizing flows to model
flexible conditional distributions and a variational distribution over DAGs.

**Algorithm:**

- Initialize edge logits Z[i,j] and per-variable MLPs (location networks)
- Sample soft adjacency: A[i,j] = sigmoid(Z[i,j] / tau) with temperature annealing
- For each target j: predict location f_j = MLP_j(masked input by A[:,j])
- Compute log-likelihood using Gaussian noise model: log p(x_j | pa(j))
- Add KL divergence for edge distribution and NOTEARS acyclicity penalty
- Optimize ELBO via gradient descent on Z and MLP weights
- Threshold final edge probabilities to get DAG

Reference: Geffner et al. (2022), "Deep End-to-end Causal Inference", arXiv.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

