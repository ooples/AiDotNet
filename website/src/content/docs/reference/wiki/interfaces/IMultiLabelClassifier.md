---
title: "IMultiLabelClassifier<T>"
description: "Interface for multi-label classification models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for multi-label classification models.

## For Beginners

Multi-label classification assigns multiple labels to each sample,
unlike traditional classification which assigns exactly one label. For example, a news article
might be tagged with both "politics" and "economy".

## How It Works

**Key differences from multi-class classification:**

- Each sample can have zero, one, or multiple labels
- Labels are not mutually exclusive
- Output is a binary indicator matrix (1 = label present, 0 = absent)

**Common approaches:**

- **Problem Transformation:** Convert to multiple binary problems (Binary Relevance, Classifier Chains)
- **Algorithm Adaptation:** Adapt algorithms to handle multiple labels (ML-kNN, RAkEL)

## Properties

| Property | Summary |
|:-----|:--------|
| `LabelNames` | Gets the label names if available. |
| `NumLabels` | Gets the number of possible labels. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Predict(Matrix<>)` | Predicts binary label indicators for input samples. |
| `PredictProbabilities(Matrix<>)` | Predicts label probabilities for input samples. |
| `Train(Matrix<>,Matrix<>)` | Trains the multi-label classifier. |

