---
title: "FlexPACBayesAlgorithm<T, TInput, TOutput>"
description: "Implementation of Flex-PAC-Bayes: Flexible PAC-Bayesian Meta-Learning with data-dependent prior construction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Flex-PAC-Bayes: Flexible PAC-Bayesian Meta-Learning with
data-dependent prior construction.

## How It Works

Flex-PAC-Bayes constructs a data-dependent prior by partially adapting the meta-parameters
on the support set (first K_prior steps), then continues adaptation with prior regularization
(K_bound steps). The "flex" parameter λ interpolates between PAC-Bayes (λ=1) and ERM (λ→0).
The prior data fraction f controls how much of the adaptation budget is allocated to prior
construction vs. posterior refinement.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputePointKL(Vector<>,Vector<>)` | Computes KL(Q \|\| P) for point posteriors using the learned prior variance: KL = 0.5 * Σ_d (θ_post_d - θ_prior_d)² / σ²_d |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_priorLogVar` | Meta-learned prior log-variance (per-parameter). |

