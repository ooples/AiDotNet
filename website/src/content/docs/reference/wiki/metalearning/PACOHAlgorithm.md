---
title: "PACOHAlgorithm<T, TInput, TOutput>"
description: "Implementation of PACOH: PAC-Bayesian Meta-Learning with Optimal Hyperparameters (Rothfuss et al., ICLR 2021)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of PACOH: PAC-Bayesian Meta-Learning with Optimal Hyperparameters
(Rothfuss et al., ICLR 2021).

## How It Works

PACOH meta-learns a Gaussian prior distribution N(μ_prior, diag(σ²_prior)) over model
parameters such that the PAC-Bayesian generalization bound is minimized. The key insight
is that optimizing the prior (not just the posterior) yields tighter generalization
guarantees for meta-learning.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAggregateKL(List<Vector<>>)` | Computes KL divergence between the empirical posterior (average of adapted params) and the prior: KL(Q \|\| P) for diagonal Gaussians. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_priorLogVar` | Prior log-variance (meta-learned). |
| `_priorMean` | Prior mean (meta-learned). |

