---
title: "VeRAAdapter<T>"
description: "VeRA (Vector-based Random Matrix Adaptation) adapter - an extreme parameter-efficient variant of LoRA."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

VeRA (Vector-based Random Matrix Adaptation) adapter - an extreme parameter-efficient variant of LoRA.

## For Beginners

VeRA is an ultra-efficient version of LoRA for extreme memory constraints.

Think of the difference this way:

- Standard LoRA: Each layer has its own pair of small matrices (A and B) that are trained
- VeRA: ALL layers share the same random matrices (A and B) which are frozen. Only tiny

scaling vectors are trained per layer.

Example parameter comparison for a 1000x1000 layer with rank=8:

- Full fine-tuning: 1,000,000 parameters
- Standard LoRA (rank=8): 16,000 parameters (98.4% reduction)
- VeRA (rank=8): ~1,600 parameters (99.84% reduction) - 10x fewer than LoRA!

Trade-offs:

- ✅ Extreme parameter efficiency (10x fewer than LoRA)
- ✅ Very low memory footprint
- ✅ Shared matrices reduce storage when adapting many layers
- ⚠️ Slightly less flexible than standard LoRA (shared random projection)
- ⚠️ Performance may be marginally lower than LoRA in some cases

When to use VeRA:

- Extreme memory constraints (mobile, edge devices)
- Fine-tuning many layers with limited resources
- Rapid prototyping with minimal parameter overhead
- When LoRA is still too expensive

## How It Works

VeRA achieves 10x fewer trainable parameters than standard LoRA by:

- Using a single pair of random low-rank matrices (A and B) shared across ALL layers
- Freezing these shared matrices (they are never trained)
- Training only small scaling vectors (d and b) that are specific to each layer

The forward computation is: output = base_layer(input) + d * (B * A * input) * b
where d and b are trainable vectors, and A and B are frozen shared matrices.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new VeRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured VeRAAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VeRAAdapter(ILayer<>,Int32,Double,Boolean)` | Initializes a new VeRA adapter wrapping an existing layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AreSharedMatricesInitialized` | Gets whether the shared matrices have been initialized. |
| `ParameterCount` | Gets the total number of trainable parameters (only the scaling vectors d and b). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateLoRALayer(Int32,Double)` | Creates a VeRA-specific layer (not used since VeRA doesn't use LoRALayer). |
| `Forward(Tensor<>)` | Performs the forward pass through the VeRA adapter. |
| `GetParameters` | Gets the current parameters as a vector (scaling vectors only). |
| `InitializeSharedMatrices(Int32,Int32,Int32,Nullable<Int32>)` | Initializes the shared random matrices used by all VeRA adapters. |
| `MergeToOriginalLayer` | Merges the VeRA adaptation into the base layer and returns the merged layer. |
| `ResetSharedMatrices` | Resets the shared matrices (useful for testing or reinitializing). |
| `ResetState` | Resets the internal state of the VeRA adapter. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `UpdateParameterGradientsFromVectors` | Updates the parameter gradients vector from the scaling vector gradients. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |
| `UpdateParametersFromLayers` | Updates the parameter vector from the current layer states. |
| `UpdateParametersFromVectors` | Updates the parameter vector from the current scaling vector values. |
| `UpdateVectorsFromParameters` | Updates the scaling vectors from the parameter vector. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_initLock` | Lock object for thread-safe shared matrix initialization. |
| `_lastInput` | Stored input from the forward pass, needed for gradient computation. |
| `_lastIntermediate` | Stored intermediate value (B * A * input) from forward pass, needed for backward pass. |
| `_scalingVectorB` | Scaling vector b (rank) - trainable per-layer parameter. |
| `_scalingVectorBGradient` | Gradient for scaling vector b computed during backpropagation. |
| `_scalingVectorD` | Scaling vector d (outputSize) - trainable per-layer parameter. |
| `_scalingVectorDGradient` | Gradient for scaling vector d computed during backpropagation. |
| `_sharedMatrixA` | Shared frozen random matrix A (inputSize × rank) used by all VeRA adapters. |
| `_sharedMatrixB` | Shared frozen random matrix B (rank × outputSize) used by all VeRA adapters. |

