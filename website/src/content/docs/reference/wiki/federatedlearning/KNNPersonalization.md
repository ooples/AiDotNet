---
title: "KNNPersonalization<T>"
description: "Implements kNN-Per — kNN-based personalization at inference time with zero extra training cost."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements kNN-Per — kNN-based personalization at inference time with zero extra training cost.

## For Beginners

kNN-Per is the simplest way to personalize a federated model:
after FL training, each client builds a small cache of (feature, label) pairs from their
local data using the global model's feature extractor. At inference time, the global model's
prediction is combined with a kNN lookup in this local cache. If the test input is similar
to local training examples, the kNN component dominates; if it's novel, the global model
dominates. This adds zero extra training cost — just a one-time cache construction.

## How It Works

Prediction:

where lambda is tuned by cross-validation or set heuristically.

Reference: Marfoq, O., et al. (2023). "kNN-Per: Nearest Neighbor-Based
Personalization for Federated Learning." ICML 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KNNPersonalization(Int32,Double)` | Creates a new kNN-Per personalization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CacheSize` | Gets the current cache size. |
| `K` | Gets the number of neighbors (k). |
| `Lambda` | Gets the kNN mixing weight (lambda). |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildCache([][],Int32[])` | Builds the local feature cache from training data. |
| `CombinedPredict([],Double[],Int32)` | Combines kNN prediction with global model prediction per the kNN-Per formula: p_final = lambda * p_kNN + (1 - lambda) * p_global. |
| `KNNPredict([],Int32)` | Performs distance-weighted kNN lookup using cosine similarity. |

