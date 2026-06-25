---
title: "ContextEncoder<T>"
description: "Context encoder for TabR that processes retrieved neighbors into a context representation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Context encoder for TabR that processes retrieved neighbors into a context representation.

## For Beginners

The context encoder does three things:

1. Takes the similar samples found by retrieval
2. Weights them by how similar they are (attention)
3. Creates a single "context" representation that summarizes the neighbors

This context is then combined with the query to make the final prediction.

## How It Works

The context encoder takes retrieved neighbors (values and labels) and combines them
using attention-weighted aggregation and cross-attention with the query.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContextEncoder(Int32,Int32,Int32)` | Initializes the context encoder. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total parameter count. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>,RetrievalContext<>)` | Encodes the retrieved context using cross-attention. |
| `ResetState` | Resets internal state. |
| `UpdateParameters()` | Updates parameters. |

