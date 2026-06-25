---
title: "MappedRandomForestModel<T>"
description: "Wrapper model that applies feature mapping before prediction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TransferLearning.Algorithms`

Wrapper model that applies feature mapping before prediction.

## For Beginners

This wrapper adapts a Random Forest trained on one set of
features to work with a different set of features. It automatically maps the input
features from the target domain to match what the source model expects, then runs
the prediction. This is how transfer learning works with tree-based models.

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `GetActiveFeatureIndices` |  |
| `GetFeatureImportance` |  |
| `GetParameters` |  |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` | Loads the mapped Random Forest model's state from a stream. |
| `Predict(Matrix<>)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` | Saves the mapped Random Forest model's current state to a stream. |
| `Serialize` |  |
| `WithParameters(Vector<>)` |  |

