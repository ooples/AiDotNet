---
title: "TabMEnsembleLayer<T>"
description: "TabM ensemble MLP (Gorishniy et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

TabM ensemble MLP (Gorishniy et al. 2024, "TabM: Advancing Tabular Deep Learning with
Parameter-Efficient Ensembling"): an MLP whose linear layers are `BatchEnsembleLayer`
(k members sharing one weight matrix via per-member rank-1 r/s adapters). The input is tiled once
across the k members, each member runs through the full MLP, and the per-member predictions are
averaged into a single output.

## How It Works

Implemented as one composite layer (held in a model's `Layers` list like any other layer).
The batch is expanded to `[batch * k, .]` once by the first BatchEnsemble layer; subsequent
layers use `Tensor{` so the member axis persists without
re-tiling, and the final layer's `Tensor{` collapses it.
All sub-layers are registered via `ILayer{` and the forward is
all tape-recorded Engine ops, so gradients flow to every member's adapters and the shared weights.
Feature count adapts to the fed input width via a rebuild. Output: `[batch, outputDim]`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabMEnsembleLayer(Int32,Int32[],Int32,Int32)` | Initializes a new `TabMEnsembleLayer`. |

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
| `UpdateParameters()` |  |

