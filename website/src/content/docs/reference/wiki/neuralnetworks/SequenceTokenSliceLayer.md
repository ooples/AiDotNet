---
title: "SequenceTokenSliceLayer<T>"
description: "Collapses a transformer encoder's `[batch, seq, dim]` hidden states down to `[batch, dim]` by selecting a single position (last, first, or a fixed middle index)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Collapses a transformer encoder's `[batch, seq, dim]` hidden
states down to `[batch, dim]` by selecting a single position
(last, first, or a fixed middle index). Used by
`TransformerArchitecture{`
when the architecture's
`SequencePooling` is
`LastToken` or
`ClsToken`.

## How It Works

The reason this is a dedicated layer (instead of inlining a
`TensorSliceAxis` call) is so the slice participates correctly
in the layer chain's shape resolution + serialization passes, and
so it can be replaced piecewise via custom-layer overrides on
`Layers`.

**Position semantics:**

- `LastToken` selects index `seq - 1` at runtime — the

actual prefix length, NOT a baked-in maxSequenceLength.

- `ClsToken` selects index 0 — the prepended summary token.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SequenceTokenSliceLayer(SequenceTokenSliceLayer<>.Position)` | Creates a slice layer that selects a single position from the sequence axis (axis 1 of a rank-3 input). |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` |  |
| `GetParameters` |  |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |
| `UpdateParameters(Vector<>)` |  |

