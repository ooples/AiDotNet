---
title: "BMAMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of BMAML: Bayesian Model-Agnostic Meta-Learning (Yoon et al., NeurIPS 2018)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of BMAML: Bayesian Model-Agnostic Meta-Learning
(Yoon et al., NeurIPS 2018).

## How It Works

BMAML uses Stein Variational Gradient Descent (SVGD) to maintain a particle ensemble
{θ¹, ..., θᴹ} that approximates the posterior distribution over task-adapted parameters.
Each particle is updated using both the task loss gradient and a repulsive kernel term
that encourages diversity among particles.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeMedianBandwidth(Vector<>[])` | Computes the RBF kernel bandwidth using the median heuristic: h² = median(pairwise squared distances) / log(M). |
| `ComputeSVGDUpdate(Vector<>[],Vector<>[],Int32,Double)` | Computes the SVGD update direction for particle i: φ(θ_i) = (1/M) Σ_j [k(θ_j, θ_i) * ∇ log p(D\|θ_j) + α * ∇_{θ_j} k(θ_j, θ_i)] |
| `InitializeParticles(Vector<>)` | Initializes M particles near the meta-parameters with Gaussian perturbation. |
| `MetaTrain(TaskBatch<,,>)` |  |

