---
title: "DiBSAlgorithm<T>"
description: "DiBS — Differentiable Bayesian Structure Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.Bayesian`

DiBS — Differentiable Bayesian Structure Learning.

## For Beginners

DiBS uses gradient-based optimization (like training a neural network)
to find not just one graph but a whole set of plausible graphs, giving you uncertainty
estimates about the causal structure.

## How It Works

DiBS uses Stein Variational Gradient Descent (SVGD) to maintain a set of particles,
each representing a possible DAG structure via continuous edge logits. The particles
are jointly optimized to approximate the posterior distribution over DAGs using
a differentiable relaxation of the acyclicity constraint.

**Algorithm:**

- Initialize K particles, each with edge logits Z_k[i,j] (d x d matrix)
- For each particle, compute edge probabilities via sigmoid(Z_k / tau)
- Compute log-posterior gradient: data likelihood + prior + acyclicity penalty
- Compute SVGD kernel: k(Z_k, Z_l) = exp(-||Z_k - Z_l||^2 / (2*bandwidth^2))
- Update particles: Z_k += lr * (1/K) * sum_l [k(Z_l, Z_k) * grad_l + grad_k(Z_l, Z_k)]
- Anneal temperature tau
- Average edge probabilities across particles for final output

Reference: Lorch et al. (2021), "DiBS: Differentiable Bayesian Structure Learning", NeurIPS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

