---
title: "LEOModel<T, TInput, TOutput>"
description: "LEO model for few-shot classification with latent space optimization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Models`

LEO model for few-shot classification with latent space optimization.

## For Beginners

After LEO adapts to a new task by optimizing
in latent space, this model stores:

## How It Works

This model stores the adapted state of LEO after latent space optimization.
It contains the feature encoder, adapted classifier parameters, and the
optimized latent code.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LEOModel(IFullModel<,,>,Vector<>,Vector<>,LEOOptions<,,>)` | Initializes a new instance of the LEOModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassifierParams` | Gets the adapted classifier parameters. |
| `LatentCode` | Gets the optimized latent code. |
| `LatentDimension` | Gets the latent dimension. |
| `Metadata` |  |
| `NumClasses` | Gets the number of classes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLogits(Vector<>)` | Computes logits from embeddings using classifier parameters. |
| `ConvertToOutput(Vector<>)` | Converts logits to the expected output type. |
| `ExtractEmbeddings()` | Extracts embeddings from input using the feature encoder. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `Predict()` |  |
| `Train(,)` |  |
| `UpdateParameters(Vector<>)` |  |

