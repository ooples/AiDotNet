---
title: "MCSLAlgorithm<T>"
description: "MCSL — Masked Gradient-Based Causal Structure Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

MCSL — Masked Gradient-Based Causal Structure Learning.

## For Beginners

MCSL adds a clever trick on top of NOTEARS. Instead of learning edge
weights directly and then thresholding, it learns a separate "switch" for each edge (on/off)
along with the weight. This makes it easier for the algorithm to decide which edges should
exist vs. not exist, leading to sparser and often more accurate graphs.

## How It Works

MCSL learns causal structure by maintaining a separate mask matrix M alongside the
weight matrix W. The effective adjacency is W * sigmoid(M/tau), where tau is a
temperature parameter that anneals from soft to hard masks. This separation of
structure (M) and weights (W) enables cleaner sparsity than L1 alone.

**Algorithm:**

- Initialize W from pairwise OLS and M (mask logits) = 0
- Compute soft mask: mask = sigmoid(M / temperature)
- Effective adjacency: W_eff = W * mask (element-wise)
- Compute L2 loss on W_eff and NOTEARS acyclicity constraint
- Update W and M via gradient descent with augmented Lagrangian
- Anneal temperature: decrease over iterations (soft → hard mask)
- Threshold final W * sigmoid(M / tau_final)

Reference: Ng et al. (2021), "Masked Gradient-Based Causal Structure Learning", SDM.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MCSLAlgorithm(CausalDiscoveryOptions)` | Initializes MCSL with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

