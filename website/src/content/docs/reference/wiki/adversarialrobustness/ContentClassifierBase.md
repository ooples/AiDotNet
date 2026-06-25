---
title: "ContentClassifierBase<T>"
description: "Base class for ML-based content classifiers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.AdversarialRobustness.Safety`

Base class for ML-based content classifiers.

## For Beginners

This is a template that makes it easier to build
different types of content classifiers. It handles the common tasks like
comparing scores to thresholds and formatting results, so you can focus
on the actual classification logic in your subclass.

## How It Works

This abstract class provides common functionality for content classifiers,
including threshold-based filtering, category management, and result formatting.
Subclasses implement the actual ML model for classification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContentClassifierBase(Double,String[])` | Initializes a new instance of the content classifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectionThreshold` | The detection threshold for classifying content as harmful. |
| `Engine` | Gets the hardware-accelerated computation engine for vectorized operations. |
| `SupportedCategories` | The supported content categories for this classifier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Vector<>)` |  |
| `ClassifyBatch(Matrix<>)` |  |
| `ClassifyText(String)` |  |
| `CreateResultFromScores(Dictionary<String,>)` | Creates a classification result from category scores. |
| `Deserialize(Byte[])` |  |
| `GetDynamicShapeInfo` |  |
| `GetInputShape` |  |
| `GetOutputShape` |  |
| `GetSupportedCategories` |  |
| `IsReady` |  |
| `LoadModel(String)` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `TextToVector(String)` | Converts text to a vector representation for classification. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultCategories` | The default categories for content classification. |
| `NumOps` | Numeric operations for type T. |

