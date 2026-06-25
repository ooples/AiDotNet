---
title: "BayProNetAlgorithm<T, TInput, TOutput>"
description: "Implementation of BayProNet: Bayesian Prototypical Networks with uncertainty-aware prototype distributions for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of BayProNet: Bayesian Prototypical Networks with uncertainty-aware
prototype distributions for few-shot learning.

## How It Works

BayProNet extends Prototypical Networks by modeling class prototypes as Gaussian
distributions rather than point estimates. In the parameter-space formulation,
adaptation produces a posterior distribution N(θ_adapted, σ²) over parameters,
and predictions are made by ensembling over parameter samples. The variance σ² is
meta-learned per parameter dimension, enabling uncertainty-aware few-shot learning.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAdaptedKL(Vector<>,Vector<>)` | KL(N(μ_post, σ²_post) \|\| N(μ_prior, I)) = 0.5 * Σ_d [σ²_d + (μ_post_d - μ_prior_d)² - 1 - log(σ²_d)] |
| `MetaTrain(TaskBatch<,,>)` |  |
| `SamplePosterior(Vector<>)` | Samples parameters from the posterior: θ_s ~ N(θ_adapted, diag(σ²)). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_posteriorLogVar` | Meta-learned per-parameter log-variance for the posterior distribution. |

