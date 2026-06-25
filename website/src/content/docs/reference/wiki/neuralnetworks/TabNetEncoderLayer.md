---
title: "TabNetEncoderLayer<T>"
description: "TabNet encoder (Arik & Pfister 2019): a sequential-attention block that produces an aggregated decision representation through several decision steps, each of which selects a sparse subset of features via a learnable mask."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

TabNet encoder (Arik & Pfister 2019): a sequential-attention block that produces an
aggregated decision representation through several decision steps, each of which selects a
sparse subset of features via a learnable mask.

## How It Works

TabNet is not a feed-forward stack, so the decision-step loop is encapsulated in this single
composite layer (held in a model's `Layers` list like any other layer). Each decision step:

The feature count is resolved lazily from the first forward input (like `DenseLayer`),
so the layer adapts to the actual fed input width. All sub-layers are registered via
`ILayer{` so the trainable-parameter walk reaches them and the
optimizer updates them; the forward pass is built entirely from tape-recorded Engine ops and
sub-layer Forward calls, so gradients flow end-to-end. The decision/attention split uses a
constant selection-matrix matmul (tape-safe) rather than a tensor slice. Output: `[batch, decisionDim]`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabNetEncoderLayer(Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32,Double,Double)` | Initializes a new `TabNetEncoderLayer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` |  |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` |  |
| `UpdateParameters()` |  |

