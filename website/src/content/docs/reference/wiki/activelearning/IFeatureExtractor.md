---
title: "IFeatureExtractor<T, TInput>"
description: "Interface for models that can extract feature representations from inputs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for models that can extract feature representations from inputs.

## For Beginners

Feature extraction converts raw input data into a
numerical representation (feature vector) that captures the important characteristics
of the input. This is useful for tasks like clustering, similarity comparisons,
and active learning strategies that need to measure diversity.

## How It Works

**Common Uses:**

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` | Gets the dimensionality of the extracted feature vectors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractFeatures()` | Extracts feature representations from the given input. |

