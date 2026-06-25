---
title: "FoldResult<T, TInput, TOutput>"
description: "Represents the results of a single fold in cross-validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the results of a single fold in cross-validation.

## For Beginners

A FoldResult contains all the performance metrics for one "fold"
in cross-validation. Think of it like a report card for a single test of your model,
where the model was trained on one subset of your data and tested on another.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FoldResult(Int32,Vector<>,Vector<>,Vector<>,Vector<>,Dictionary<String,>,Nullable<TimeSpan>,Nullable<TimeSpan>,Int32,IFullModel<,,>,ClusteringMetrics<>,Int32[],Int32[])` | Creates a new instance of the FoldResult class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualValues` | Gets the actual values from the validation dataset. |
| `ClusteringMetrics` | Gets the clustering quality metrics for this fold, if applicable. |
| `EvaluationTime` | Gets the time taken to evaluate the model for this fold. |
| `FeatureImportance` | Gets the feature importance scores for this fold. |
| `FoldIndex` | Gets the index of this fold in the cross-validation process. |
| `Model` | Gets the trained model instance for this fold. |
| `PredictedValues` | Gets the predicted values for the validation dataset. |
| `TrainingErrors` | Gets the error statistics for the training data. |
| `TrainingIndices` | Gets the indices of the training samples in this fold. |
| `TrainingTime` | Gets the time taken to train the model for this fold. |
| `ValidationErrors` | Gets the error statistics for the validation data. |
| `ValidationIndices` | Gets the indices of the validation samples in this fold. |
| `ValidationPredictionStats` | Gets the prediction statistics for the validation data. |

