---
title: "MetaDMAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-DM: Applications of Diffusion Models on Few-Shot Learning (Hu et al., ICIP 2024)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-DM: Applications of Diffusion Models on Few-Shot Learning
(Hu et al., ICIP 2024).

## How It Works

Meta-DM uses a DDPM-style diffusion model as a data augmentation module for few-shot learning.
Rather than replacing the meta-learning algorithm, it augments the support set by generating
synthetic samples conditioned on existing few-shot data, then performs standard gradient-based
adaptation on the enriched dataset.

**Algorithm:**

**Key advantage:** Modular — can be composed with any gradient-based meta-learner.
The diffusion-based augmentation enriches the support set, reducing overfitting.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeDistributionMatchingLoss(Vector<>,Double[][])` | Computes distribution matching loss between real and synthetic feature distributions (moment matching: mean and variance). |
| `ComputePrototype(Vector<>)` | Computes a class prototype from a feature vector. |
| `GenerateSyntheticFeatures(Vector<>,Int32)` | Generates synthetic features by running the reverse diffusion process conditioned on the class prototype. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `ModulateWithSynthetic(Vector<>,Double[][],Vector<>)` | Modulates the real gradient with information from synthetic features to create an augmented gradient that benefits from the diffusion-generated data. |
| `PredictFeatureNoise(Double[],Vector<>,Int32)` | Feature denoiser MLP: predicts noise given noised features, prototype condition, and timestep. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_betas` | Noise schedule. |
| `_denoiserParams` | Feature denoiser parameters for generating synthetic features. |

