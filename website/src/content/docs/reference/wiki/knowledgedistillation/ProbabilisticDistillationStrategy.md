---
title: "ProbabilisticDistillationStrategy<T>"
description: "Probabilistic distillation that transfers distributional knowledge by matching statistical properties."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Strategies`

Probabilistic distillation that transfers distributional knowledge by matching statistical properties.

## How It Works

**For Production Use:** This strategy treats model outputs as samples from probability distributions
and transfers knowledge about the entire distribution, not just point predictions. It matches statistical
moments (mean, variance, higher moments) and can use measures like Maximum Mean Discrepancy (MMD).

**Key Concept:** Standard distillation matches individual predictions, but neural networks can be
viewed as probabilistic models. This strategy captures uncertainty and distribution shape by matching:

1. First moment (mean) - Expected predictions
2. Second moment (variance) - Prediction uncertainty
3. Distribution distance (MMD, Wasserstein) - Overall shape

**Implementation:** We provide three modes:

- MomentMatching: Match mean and variance of predictions across batch
- MaximumMeanDiscrepancy: Use MMD with RBF kernel to match distributions
- EntropyTransfer: Match prediction entropy (uncertainty calibration)

**Research Basis:** Based on probabilistic knowledge distillation and Bayesian neural networks.
Particularly useful for uncertainty quantification and ensemble distillation.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistributionalLoss(Vector<>[],Vector<>[])` | Computes distributional loss by matching statistical properties across a batch. |

