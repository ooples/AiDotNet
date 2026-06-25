---
title: "UncertaintyPredictionResult<T, TOutput>"
description: "Represents a prediction result augmented with uncertainty information."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents a prediction result augmented with uncertainty information.

## For Beginners

This lets you ask the model both:

- "What is the prediction?"
- "How uncertain is that prediction?"

## How It Works

This type is returned by the facade method `AiModelResult.PredictWithUncertainty(...)`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UncertaintyPredictionResult(UncertaintyQuantificationMethod,,,IReadOnlyDictionary<String,Tensor<>>,RegressionConformalInterval<>,ClassificationConformalPredictionSet)` | Initializes a new instance of the `UncertaintyPredictionResult` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassificationSet` | Gets an optional conformal classification prediction set, when configured and supported. |
| `MethodUsed` | Gets the uncertainty method that was used to produce this result. |
| `Metrics` | Gets additional uncertainty diagnostics. |
| `Prediction` | Gets the point prediction (mean / expected prediction). |
| `RegressionInterval` | Gets an optional conformal regression interval, when configured and supported. |
| `Variance` | Gets the per-output predictive variance (when available). |

