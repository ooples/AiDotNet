---
title: "DiversityStrategyBase<T, TInput, TOutput>"
description: "Base class for diversity strategies with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ActiveLearning.Interfaces`

Base class for diversity strategies with common functionality.

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistance(Vector<>,Vector<>)` | Computes Euclidean distance between two feature vectors. |
| `ComputeDistanceMatrix(IDataset<,,>)` |  |
| `ComputeDiversityScores(IDataset<,,>,IDataset<,,>)` |  |
| `GetFeatureRepresentation()` |  |
| `SelectDiverseSamples(IDataset<,,>,IDataset<,,>,Int32)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Gets numeric operations helper. |

