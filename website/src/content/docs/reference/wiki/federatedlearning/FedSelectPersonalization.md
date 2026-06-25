---
title: "FedSelectPersonalization<T>"
description: "Implements FedSelect — learned sparse binary masks for personalization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements FedSelect — learned sparse binary masks for personalization.

## For Beginners

Instead of deciding up-front which layers to personalize,
FedSelect learns a binary mask for each client that determines, for each parameter,
whether it should be shared (aggregated globally) or personalized (kept local). The mask
itself is learned during training using straight-through estimator gradients. This gives
each client a different "personalization pattern" that best fits their data.

## How It Works

Algorithm:

Reference: FedSelect: Personalizing FL with Learned Parameter Selection (2023).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedSelectPersonalization(Double,Double)` | Creates a new FedSelect personalization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaskRegularization` | Gets the mask regularization weight. |
| `MaskThreshold` | Gets the mask binarization threshold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeRegularizationLoss` | Computes the L1 regularization loss on the mask to encourage sparsity (fewer personalized params). |
| `ExtractSharedParameters(Dictionary<String,[]>)` | Extracts shared parameters (where mask is 0) for aggregation. |
| `GetPersonalizationRatio` | Computes the mask sparsity (fraction of parameters that are personalized). |
| `InitializeMasks(Dictionary<String,[]>,Double)` | Initializes mask logits for a model structure. |
| `UpdateMask(String,[],[],[],Double)` | Updates mask logits using straight-through estimator (STE) gradients. |

