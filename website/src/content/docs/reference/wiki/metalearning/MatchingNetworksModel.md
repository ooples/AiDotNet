---
title: "MatchingNetworksModel<T, TInput, TOutput>"
description: "Matching Networks model for inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Matching Networks model for inference.

## How It Works

This model encapsulates the Matching Networks inference mechanism with pre-computed
support embeddings. It is returned by `IMetaLearningTask{`
and provides fast classification using attention over support examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MatchingNetworksModel(IFullModel<,,>,,,MatchingNetworksOptions<,,>,INumericOperations<>)` | Initializes a new instance of the MatchingNetworksModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` | Gets the model metadata. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetModelMetadata` | Gets model metadata. |
| `GetParameters` | Gets encoder parameters. |
| `Predict()` | Makes predictions using attention over support examples. |
| `Train(,)` | Training is not supported for inference models. |
| `UpdateParameters(Vector<>)` | Parameter updates are not supported for inference models. |

