---
title: "DiffWaveNetwork<T>"
description: "DiffWave neural network with dilated convolutions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Audio`

DiffWave neural network with dilated convolutions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffWaveNetwork(IEngine,Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new DiffWaveNetwork. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the number of parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>,Int32,Tensor<>)` | Forward pass through the network — Kong et al. |
| `GetParameters` | Gets all parameters. |
| `ResolveLayerShapesFor(Int32[])` | Replays the lazy shape-resolution pass that would happen on the first `Tensor{` with the given audio shape, without keeping the dummy output. |
| `SetParameters(Vector<>)` | Sets all parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_engine` | The engine instance used for all tape-tracked tensor ops in this network and its residual blocks. |

