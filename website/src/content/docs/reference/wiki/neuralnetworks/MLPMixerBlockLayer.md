---
title: "MLPMixerBlockLayer<T>"
description: "A single MLP-Mixer block: temporal-axis MLP + channel-axis MLP, each wrapped with pre-norm and residual, per Tolstikhin et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A single MLP-Mixer block: temporal-axis MLP + channel-axis MLP, each wrapped with
pre-norm and residual, per Tolstikhin et al. 2021 "MLP-Mixer: An all-MLP Architecture
for Vision" (extended to time-series patches by Ekambaram et al. 2024 for Tiny Time
Mixers).

## How It Works

Operates on input tensors of shape `[B, numPatches, hiddenDim]`. The forward sequence is:

The temporal mixer mixes information across patches (time dimension); the channel
mixer mixes information across hidden features at each patch position. The transpose
is required so a plain `DenseLayer` (which operates on the last axis)
can be applied across the patch axis.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MLPMixerBlockLayer(Int32,Int32,Int32)` | Initializes a new `MLPMixerBlockLayer`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Persists the constructor arguments so the deserializer can rebuild this layer at the same shape. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateParameters()` |  |

