---
title: "CLSToken<T>"
description: "CLS (Classification) Token for transformer-based tabular models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

CLS (Classification) Token for transformer-based tabular models.

## For Beginners

The CLS token serves as a "summary" position:

- It's added to the beginning of your features
- The transformer allows it to attend to all features
- After processing, the CLS token "knows about" all features
- We use its final representation for prediction

This is the same approach used in BERT for text classification.

## How It Works

The CLS token is a learnable embedding prepended to the input sequence.
After transformer processing, the CLS token's representation is used
as an aggregate representation of the entire input for classification/regression.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CLSToken(Int32,Double)` | Initializes a new CLS token with the specified embedding dimension. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `ParameterCount` | Gets the number of parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractCLS(Tensor<>)` | Extracts the CLS token representation from transformer output. |
| `GetEmbedding` | Gets the current CLS embedding values. |
| `PrependCLS(Tensor<>)` | Prepends the CLS token to the input embeddings. |
| `ResetGradients` | Resets gradients to zero. |
| `UpdateParameters()` | Updates the CLS token embedding. |

