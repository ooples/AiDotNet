---
title: "FewTUREAlgorithm<T, TInput, TOutput>"
description: "Implementation of FewTURE (Few-shot Transformer with Uncertainty and Reliable Estimation) (Hiller et al., ECCV 2022)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of FewTURE (Few-shot Transformer with Uncertainty and Reliable Estimation)
(Hiller et al., ECCV 2022).

## For Beginners

FewTURE compares images piece by piece, not as wholes:

**Standard approach:**
Represent each image as ONE feature vector, compare vectors.
Problem: A bird's beak might be the key difference, but it's a tiny part of the image.

**FewTURE's approach:**

1. Split each image into patches (like a puzzle, e.g., 14x14 = 196 patches)
2. Use a Vision Transformer to get a feature for each patch (token)
3. Compare queries to support classes at the PATCH level
4. Estimate uncertainty for each patch comparison
5. Weight reliable patches more, uncertain patches less

**Why uncertainty matters:**
Not all patches are equally informative:

- Background patches are mostly noise (high uncertainty)
- Discriminative patches (beak, stripes) are informative (low uncertainty)

FewTURE learns to focus on the informative patches automatically.

## How It Works

FewTURE uses vision transformers with token-level local features and uncertainty estimation
for few-shot classification. Instead of comparing global image features, it compares at the
patch/token level and weights predictions by their estimated reliability.

**Algorithm - FewTURE:**

Reference: Hiller, M., Ma, R., Harber, M., & Ommer, B. (2022).
Rethinking Generalization in Few-Shot Classification. ECCV 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FewTUREAlgorithm(FewTUREOptions<,,>)` | Initializes a new FewTURE meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using uncertainty-weighted predictions. |
| `ComputeUncertaintyWeights(Vector<>,Vector<>)` | Computes uncertainty-weighted classification weights for query features. |
| `EstimateUncertainty(Vector<>)` | Estimates uncertainty for a prediction using the configured method. |
| `InitializeUncertaintyModule` | Initializes the uncertainty estimation module. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_uncertaintyParams` | Parameters for the uncertainty estimation module. |

