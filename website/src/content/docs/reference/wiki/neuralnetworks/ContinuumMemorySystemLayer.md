---
title: "ContinuumMemorySystemLayer<T>"
description: "Continuum Memory System (CMS) layer for neural networks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Continuum Memory System (CMS) layer for neural networks.
Implements a sequential chain of MLP blocks with different update frequencies.
Based on Equations 30-31 from "Nested Learning" paper.
yt = MLP^(fk)(MLP^(fk-1)(...MLP^(f1)(xt)))

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContinuumMemorySystemLayer(Int32[],Int32,Int32,Int32[],[],IEngine)` | Creates a CMS layer as a chain of MLP blocks. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkSizes` | Gets the chunk sizes for gradient accumulation. |
| `ParameterCount` | Indicates whether this layer supports training. |
| `SupportsGpuExecution` | Indicates whether this layer supports GPU execution. |
| `UpdateFrequencies` | Gets the update frequencies for each level. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` | Clears all accumulated gradients across all levels. |
| `ConsolidateMemory` | Consolidates memory from faster to slower levels. |
| `ForwardGpu(Tensor<>[])` | GPU-accelerated forward pass chaining through all MLP blocks. |
| `GetMLPBlocks` | Gets the MLP blocks in the chain. |
| `GetParameterGradients` | Gets the parameter gradients for all MLP blocks. |
| `GetParameters` | Gets all parameters from all MLP blocks in the chain. |
| `ResetMemory` | Resets all MLP blocks in the chain. |
| `ResetState` | Resets the state of the layer (required by LayerBase). |
| `SetParameters(Vector<>)` | Sets all parameters for all MLP blocks in the chain. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |

