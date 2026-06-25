---
title: "BOILModel<T, TInput, TOutput>"
description: "BOIL model for few-shot classification with body-only adaptation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Models`

BOIL model for few-shot classification with body-only adaptation.

## For Beginners

After BOIL adapts to a new task by training only
the feature extractor (body) on support examples, this model stores:

## How It Works

This model stores the adapted state of BOIL after inner-loop adaptation.
It contains the adapted feature extractor (body) and the frozen classification head.

When making predictions, the model extracts features using the adapted body
and classifies using the frozen head. This is the opposite of ANIL which
freezes the body and adapts the head.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BOILModel(IFullModel<,,>,Vector<>,Vector<>,Vector<>,BOILOptions<,,>)` | Initializes a new instance of the BOILModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptedBodyParams` | Gets the adapted body parameters. |
| `FeatureDimension` | Gets the feature dimension expected by the head. |
| `HeadBias` | Gets the frozen head bias (may be null if not used). |
| `HeadWeights` | Gets the frozen head weights. |
| `Metadata` |  |
| `NumClasses` | Gets the number of classes this model is adapted for. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAdaptedBodyParameters` | Applies the adapted body parameters to the base model. |
| `ComputeLogits(Vector<>)` | Computes logits from features using frozen head parameters. |
| `ConvertToOutput(Vector<>)` | Converts logits to the expected output type. |
| `ExtractFeatures()` | Extracts features from input using the adapted body. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `Predict()` |  |
| `Train(,)` |  |
| `UpdateParameters(Vector<>)` |  |

