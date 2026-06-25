---
title: "MetaOptNetModel<T, TInput, TOutput>"
description: "MetaOptNet model for few-shot classification with convex optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Models`

MetaOptNet model for few-shot classification with convex optimization.

## For Beginners

After MetaOptNet sees the support examples and
solves for the optimal classifier, this model stores:

## How It Works

This model stores the adapted state of MetaOptNet after solving the convex
optimization problem on the support set.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaOptNetModel(IFullModel<,,>,Matrix<>,,MetaOptNetOptions<,,>)` | Initializes a new instance of the MetaOptNetModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassifierWeights` | Gets the classifier weights. |
| `Metadata` |  |
| `NumClasses` | Gets the number of classes. |
| `Temperature` | Gets the temperature parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogits(Matrix<>)` | Computes logits using classifier weights. |
| `ConvertToOutput(Vector<>)` | Converts logits to the expected output type. |
| `ExtractEmbeddings()` | Extracts embeddings from input using the feature encoder. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `NormalizeEmbeddings(Matrix<>)` | Normalizes embeddings to unit norm. |
| `Predict()` |  |
| `ScaleByTemperature(Vector<>)` | Scales logits by temperature. |
| `Train(,)` |  |
| `UpdateParameters(Vector<>)` |  |

