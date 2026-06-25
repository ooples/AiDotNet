---
title: "ANILModel<T, TInput, TOutput>"
description: "ANIL model for few-shot classification with head-only adaptation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Models`

ANIL model for few-shot classification with head-only adaptation.

## For Beginners

After ANIL adapts to a new task by training only
the classification head on support examples, this model stores:

## How It Works

This model stores the adapted state of ANIL after inner-loop adaptation.
It contains the frozen feature extractor (body) and the adapted classification head.

When making predictions, the model extracts features using the frozen body
and classifies using the adapted head.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ANILModel(IFullModel<,,>,Vector<>,Vector<>,ANILOptions<,,>)` | Initializes a new instance of the ANILModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FeatureDimension` | Gets the feature dimension expected by the head. |
| `HeadBias` | Gets the adapted head bias (may be null if not used). |
| `HeadWeights` | Gets the adapted head weights. |
| `Metadata` |  |
| `NumClasses` | Gets the number of classes this model is adapted for. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogits(Vector<>)` | Computes logits from features using adapted head parameters. |
| `ConvertToOutput(Vector<>)` | Converts logits to the expected output type. |
| `ExtractFeatures()` | Extracts features from input using the frozen body. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `Predict()` |  |
| `Train(,)` |  |
| `UpdateParameters(Vector<>)` |  |

