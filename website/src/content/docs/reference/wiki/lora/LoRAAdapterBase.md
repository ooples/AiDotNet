---
title: "LoRAAdapterBase<T>"
description: "Abstract base class for LoRA (Low-Rank Adaptation) adapters that wrap existing layers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.LoRA.Adapters`

Abstract base class for LoRA (Low-Rank Adaptation) adapters that wrap existing layers.

## For Beginners

This is the foundation for all LoRA adapters in the library.

A LoRA adapter wraps an existing layer (like a dense or convolutional layer) and adds
a small "correction layer" that learns what adjustments are needed. This base class:

- Manages both the original layer and the LoRA correction layer
- Handles parameter synchronization between them
- Provides common forward/backward pass logic (original + correction)
- Lets specialized adapters handle layer-specific details

This design allows you to create LoRA adapters for any layer type by:

1. Inheriting from this base class
2. Implementing layer-specific validation
3. Implementing how to merge the LoRA weights back into the original layer

The result is parameter-efficient fine-tuning that works across different layer architectures!

## How It Works

This base class provides common functionality for all LoRA adapter implementations.
It manages the base layer, LoRA layer, and parameter synchronization, while allowing
derived classes to implement layer-type-specific logic such as merging and validation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoRAAdapterBase(ILayer<>,Int32,Double,Boolean)` | Initializes a new LoRA adapter base with the specified parameters. |
| `LoRAAdapterBase(ILayer<>,Int32,Double,Boolean,ValueTuple<Int32[],Boolean>)` | Internal ctor that takes the resolved-input-shape result tuple from `ILayer{`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the scaling factor (alpha) for the LoRA adaptation. |
| `BaseLayer` | Gets the base layer being adapted with LoRA. |
| `IsBaseLayerFrozen` | Gets whether the base layer's parameters are frozen during training. |
| `LoRALayer` | Gets the LoRA layer providing the low-rank adaptation. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `Rank` | Gets the rank of the low-rank decomposition. |
| `SupportsTraining` | Gets whether this adapter supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BothDimsResolved(Int32,Int32)` | Contract helper for `Int32@)`: bool return is true iff BOTH dims were resolved (review #1368 C6WO4 / C6WPP / C7mmB / C7G8-). |
| `CreateMergedLayerWithClone(Vector<>)` | Helper method to create a merged layer by cloning the base layer and updating its parameters. |
| `DeclaresShapeWithoutInput(LayerBase<>)` | Exception-safe probe of the `TryDeclareShape` oracle: `true` means the layer's parameters are materialised without needing a synthetic input shape forced on it. |
| `EnsureBaseLayerShapeResolved` | Force-resolve `_baseLayer`'s lazy shape using the input dim that the LoRA layer already settled on. |
| `Forward(Tensor<>)` | Performs the forward pass through both base and LoRA layers. |
| `GetMetadata` | Persists the inner-layer type name and shape so DeserializationHelper can reconstruct the WRAPPED base layer with the right concrete type instead of the prior `DenseLayer<T>` placeholder. |
| `GetParameters` | Gets the current parameters as a vector. |
| `InferInputSizeFromWeights(ILayer<>,IReadOnlyList<Tensor<>>)` | Creates the LoRA layer for this adapter. |
| `MergeToDenseOrFullyConnected` | Merges LoRA weights into the base layer for DenseLayer or FullyConnectedLayer. |
| `MergeToOriginalLayer` | Merges the LoRA adaptation into the base layer and returns the merged layer. |
| `PackBaseAndLoraParameters` | Non-virtual pack of `_baseLayer` + `_loraLayer` parameters into `Parameters`. |
| `RebuildParametersAfterDerivedInit` | Derived adapter classes that override `ParameterCount` to include extra state (delta weights, importance scores, etc.) MUST call this method at the end of their constructor body so the base class's `Parameters` vector is re-allocated agains… |
| `ResetState` | Resets the internal state of both the base layer and LoRA layer. |
| `ResolveBaseInputShapeWithProvenance(ILayer<>)` | Returns the resolved base-input shape AND a flag indicating whether the shape is authoritative (came from the layer's own resolved shape or its actual weight matrix) vs a synthetic `outSize * 2` heuristic. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `TryInferBothDimsFromWeights(ILayer<>,IReadOnlyList<Tensor<>>,Int32,Int32)` | Try to infer BOTH input and output dimensions from the base layer's trainable weight matrix in a single pass. |
| `UpdateLayersFromParameters` | Updates the layers from the parameter vector. |
| `UpdateParameterGradientsFromLayers` | Updates the parameter gradients vector from the layer gradients. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |
| `UpdateParametersFromLayers` | Updates the parameter vector from the current base and LoRA layer states. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseLayer` | The base layer being adapted. |
| `_freezeBaseLayer` | Whether the base layer's parameters are frozen (not trainable). |
| `_loraLayer` | The LoRA layer that provides the adaptation. |

