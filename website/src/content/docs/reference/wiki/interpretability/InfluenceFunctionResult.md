---
title: "InfluenceFunctionResult<T>"
description: "Result of influence function computation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Result of influence function computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InfluenceFunctionResult(Vector<>,,Vector<>,Vector<>,)` | Initializes a new influence function result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Influences` | Gets influence scores for each training sample. |
| `Loss` | Gets the loss on the test input. |
| `NumTrainingSamples` | Gets the number of training samples. |
| `Prediction` | Gets the model's prediction on the test input. |
| `TestInput` | Gets the test input that was explained. |
| `TestLabel` | Gets the test label. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetMostHarmful(Int32)` | Gets the most harmful training samples (lowest/most negative influence). |
| `GetMostHelpful(Int32)` | Gets the most helpful training samples (highest positive influence). |
| `GetTopInfluential(Int32)` | Gets the top K most influential training samples. |
| `ToString` | Returns a human-readable summary. |

