---
title: "MixedPrecisionScope"
description: "Provides an ambient context for mixed-precision operations during forward and backward passes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MixedPrecision`

Provides an ambient context for mixed-precision operations during forward and backward passes.

## For Beginners

MixedPrecisionScope is like a "mode switch" that tells the neural network
to use lower precision (FP16) for faster computation while keeping certain operations in full precision (FP32)
for numerical stability.

The scope is automatically managed by the training loop - you don't need to create it yourself.
When inside a scope:

- Most operations use FP16 (faster, less memory)
- Certain layers (like LayerNorm, Softmax) stay in FP32 for stability
- The scope tracks both FP16 and FP32 versions of tensors so layers can access what they need

## How It Works

**Technical Details:** The scope uses a thread-static pattern to provide ambient context.
Layers can check `Current` to determine if mixed-precision is active and whether
they should use FP16 or FP32 based on the `LayerPrecisionPolicy`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MixedPrecisionScope(MixedPrecisionContext,LayerPrecisionPolicy)` | Creates a new mixed-precision scope. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Context` | Gets the mixed-precision context associated with this scope. |
| `Current` | Gets the currently active mixed-precision scope, or null if not in a scope. |
| `IsActive` | Gets whether this scope is currently active (is the current scope). |
| `Policy` | Gets the layer precision policy for this scope. |
| `RegisteredTensorCount` | Gets the number of registered tensors. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CastToFP16(Tensor<Single>)` | Converts an FP32 tensor to FP16. |
| `CastToFP32(Tensor<Half>)` | Converts an FP16 tensor back to FP32. |
| `ClearTensors` | Clears all registered tensors to free memory. |
| `Dispose` | Disposes the scope and restores the previous scope (if any). |
| `GetFP16Tensor(String)` | Retrieves the FP16 version of a previously registered tensor. |
| `GetFP32Tensor(String)` | Retrieves the FP32 version of a previously registered tensor. |
| `GetLayerPrecision(String)` | Gets the precision type to use for a specific layer. |
| `HasTensor(String)` | Checks if a tensor with the given name has been registered. |
| `RegisterAndCastToFP16(String,Tensor<Single>)` | Registers an FP32 tensor and returns its FP16 equivalent. |
| `SetCurrentScope(MixedPrecisionScope)` | Sets the current scope (used by constructor and Dispose). |
| `ShouldUseFP32(String)` | Determines if a layer should use full precision (FP32) based on the policy. |
| `ShouldUseHigherPrecision(String)` | Determines if a layer should use higher precision than the default (FP32 or FP16 when default is FP8). |
| `ToString` | Gets a string representation of the scope's current state. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_context` | The mixed-precision context providing loss scaling and weight management. |
| `_current` | Thread-static storage for the current scope, enabling ambient context access. |
| `_disposed` | Whether the scope has been disposed. |
| `_fp16Tensors` | Storage for FP16 (Half) versions of tensors. |
| `_fp32Tensors` | Storage for FP32 versions of tensors (for layers that need full precision). |
| `_policy` | The policy determining which layers use which precision. |
| `_previous` | The previous scope (for nested scope support). |

