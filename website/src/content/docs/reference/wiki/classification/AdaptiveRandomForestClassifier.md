---
title: "AdaptiveRandomForestClassifier<T>"
description: "Implements Adaptive Random Forest (ARF) for online classification with drift detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Online`

Implements Adaptive Random Forest (ARF) for online classification with drift detection.

## For Beginners

Adaptive Random Forest is an ensemble of Hoeffding trees designed
to handle concept drift in data streams. Each tree has an associated drift detector, and when
drift is detected, the affected tree is replaced with a new one trained on recent data.

## How It Works

**How it works:**

- Maintain an ensemble of Hoeffding trees
- Each tree has a "warning" detector and a "drift" detector
- When warning is triggered, start training a background tree
- When drift is confirmed, replace the old tree with the background tree
- Use majority voting for predictions

**Key innovations:**

- Per-tree drift detection for local adaptation
- Resampling with Poisson(λ=6) for diversity
- Background tree preparation for smooth transitions
- Weighted voting based on tree accuracy

**Advantages:**

- Handles both sudden and gradual drift
- Maintains diversity through random subspaces
- Provides robust predictions through ensemble voting
- Adapts locally without resetting entire ensemble

**Reference:** Gomes et al., "Adaptive Random Forests for Evolving Data Stream Classification" (2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveRandomForestClassifier(AdaptiveRandomForestOptions<>)` | Creates a new Adaptive Random Forest classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageTreeAccuracy` | Gets the average accuracy across all trees. |
| `IsWarm` | Gets whether the model has seen at least one sample. |
| `SamplesSeen` | Gets the total number of samples the model has seen. |
| `TreeCount` | Gets the number of trees in the ensemble. |
| `TreesInWarning` | Gets the number of trees currently in warning state. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` |  |
| `ComputeGradients(Matrix<>,Vector<>,ILossFunction<>)` |  |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` | Deserializes the trained ARF ensemble including all Hoeffding trees and metadata. |
| `GetOptions` |  |
| `GetParameters` |  |
| `PartialFit(Matrix<>,Vector<>)` | Updates the model with a batch of training samples. |
| `PartialFit(Vector<>,)` | Updates the model with a single training sample. |
| `Predict(Matrix<>)` |  |
| `Serialize` | Serializes the trained ARF ensemble including all Hoeffding trees and metadata. |
| `SetParameters(Vector<>)` |  |
| `Train(Matrix<>,Vector<>)` |  |
| `WithParameters(Vector<>)` |  |

