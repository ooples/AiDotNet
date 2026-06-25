---
title: "InferenceContext<T>"
description: "Provides a scoped context for inference operations with automatic tensor pooling and lifecycle management."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Memory`

Provides a scoped context for inference operations with automatic tensor pooling and lifecycle management.
All tensors rented through this context are tracked and automatically returned to the pool when disposed.

## How It Works

InferenceContext simplifies memory management during neural network inference by:

- Tracking all rented tensors automatically
- Returning all tensors to the pool on disposal (even if Release wasn't called)
- Providing convenient rent methods for common tensor shapes

Basic usage example:

For ambient context support (avoiding parameter threading), use `InferenceScope`:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InferenceContext(TensorPool<>,Int32)` | Initializes a new instance of the `InferenceContext` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsPoolingEnabled` | Gets or sets whether pooling is enabled. |
| `Pool` | Gets the underlying tensor pool used by this context. |
| `RentedTensorCount` | Gets the number of tensors currently rented from this context. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Disposes the context and returns all rented tensors to the pool. |
| `Dispose(Boolean)` | Disposes the context resources. |
| `Release(Tensor<>)` | Releases a tensor back to the pool before the context is disposed. |
| `Rent(Int32[])` | Rents a tensor with the specified shape from the pool. |
| `Rent1D(Int32)` | Rents a 1D tensor (vector) with the specified length. |
| `Rent2D(Int32,Int32)` | Rents a 2D tensor (matrix) with the specified dimensions. |
| `Rent3D(Int32,Int32,Int32)` | Rents a 3D tensor with the specified dimensions. |
| `Rent4D(Int32,Int32,Int32,Int32)` | Rents a 4D tensor with the specified dimensions (typically for image data). |
| `RentLike(Tensor<>)` | Rents a tensor with the same shape as the template tensor. |

