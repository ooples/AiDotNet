---
title: "CompositeConditioningBase<T>"
description: "Base class for conditioning modules that compose other conditioners rather than owning their own learnable weights."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion.Conditioning`

Base class for conditioning modules that compose other conditioners rather than
owning their own learnable weights. Examples: `DualTextConditioner`
and `TripleTextConditioner`, which delegate encoding work to one or
more inner CLIP / T5 / OpenCLIP encoders and then merge their outputs.

## How It Works

This base intentionally does not allocate token / position / transformer weights
(unlike `TextConditioningBase`, whose constructor sizes weight
vectors against a single transformer). Composite conditioners hold their inner
conditioners as fields and forward calls to them, so all that is shared between
composites is the boilerplate every conditioner needs:

The `Engine` property is exposed as a protected instance member to
match the convention used by the rest of AiDotNet's base classes (see
`NeuralNetworkBase`, `AdversarialAttackBase`, `AugmentationBase`,
etc.). Subclasses MUST route all engine-dispatched tensor operations through this
property rather than calling `AiDotNetEngine.Current` directly, so that
future per-instance engine overrides remain a single-point change.

## Properties

| Property | Summary |
|:-----|:--------|
| `ConditioningType` |  |
| `EmbeddingDimension` |  |
| `Engine` | Hardware-accelerated engine for vector/tensor operations. |
| `MaxSequenceLength` |  |
| `ProducesPooledOutput` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Encode(Tensor<>)` |  |
| `EncodeText(Tensor<>,Tensor<>)` |  |
| `GetPooledEmbedding(Tensor<>)` |  |
| `GetUnconditionalEmbedding(Int32)` |  |
| `Tokenize(String)` |  |
| `TokenizeBatch(String[])` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |

