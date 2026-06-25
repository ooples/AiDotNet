---
title: "ColumnEmbedding<T>"
description: "Column (positional) embedding for tabular transformers like TabTransformer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Column (positional) embedding for tabular transformers like TabTransformer.

## For Beginners

Column embeddings help the model understand:

- "This is feature #1 (age)"
- "This is feature #2 (income)"

Without column embeddings, the model wouldn't know which feature is which
after the attention layers mix them together.

## How It Works

Column embeddings provide position information to the transformer, telling it
which column/feature each embedding came from. This is analogous to positional
encodings in NLP transformers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColumnEmbedding(Int32,Int32,Boolean,Double)` | Initializes column embeddings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDim` | Gets the embedding dimension. |
| `NumColumns` | Gets the number of columns. |
| `ParameterCount` | Gets the number of parameters (0 if using sinusoidal). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddColumnEmbeddings(Tensor<>)` | Adds column embeddings to feature embeddings. |
| `GetColumnEmbedding(Int32)` | Gets the column embedding for a specific column. |
| `ResetGradients` | Zeros all stored gradients. |
| `UpdateParameters()` | Updates embeddings via gradient descent using the gradients computed by `Backward`. |

