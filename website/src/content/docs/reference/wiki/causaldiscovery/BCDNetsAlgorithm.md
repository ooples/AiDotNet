---
title: "BCDNetsAlgorithm<T>"
description: "BCD-Nets — Bayesian Causal Discovery Networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Bayesian`

BCD-Nets — Bayesian Causal Discovery Networks.

## For Beginners

BCD-Nets learn both the graph structure AND the strength of each
connection simultaneously, using modern deep learning optimization techniques. They
provide uncertainty estimates for both.

## How It Works

BCD-Nets use variational inference to approximate the joint posterior over DAG structures
and parameters. The graph structure is parameterized via Gumbel-Softmax continuous
relaxation of binary edge variables, and parameters (edge weights) are modeled via
a factorized Gaussian variational posterior. Both are optimized jointly via gradient
ascent on the ELBO.

**Algorithm:**

- Initialize edge logits Z[i,j] and weight means mu[i,j], log-variances logvar[i,j]
- Sample edges: E[i,j] ~ GumbelSigmoid(Z[i,j]/tau)
- Sample weights: W[i,j] ~ N(mu[i,j], exp(logvar[i,j]))
- Effective adjacency: A = E * W (element-wise)
- Compute ELBO = E_q[log p(X|A)] - KL(q(Z)||p(Z)) - KL(q(W)||p(W)) - lambda*h(E)
- Update Z, mu, logvar via gradient ascent
- Anneal temperature tau and update augmented Lagrangian

Reference: Cundy et al. (2021), "BCD Nets: Scalable Variational Approaches for
Bayesian Causal Discovery", NeurIPS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

