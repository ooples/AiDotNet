---
title: "PrototypeExplanation<T>"
description: "Represents the result of a Prototype-based explanation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Represents the result of a Prototype-based explanation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ContrastPrototypes` | Gets or sets prototypes with different class (contrast examples). |
| `DistanceMetric` | Gets or sets the distance metric used. |
| `DistinguishingFeatures` | Gets or sets features that distinguish from the nearest contrast prototype. |
| `FeatureDifferences` | Gets or sets feature differences from the nearest same-class prototype. |
| `FeatureNames` | Gets or sets the feature names. |
| `Input` | Gets or sets the input instance. |
| `NearestPrototypes` | Gets or sets the nearest prototypes (regardless of class). |
| `PredictedClass` | Gets or sets the predicted class. |
| `Prediction` | Gets or sets the prediction value. |
| `SameClassPrototypes` | Gets or sets prototypes with the same class (supporting evidence). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a human-readable summary. |

