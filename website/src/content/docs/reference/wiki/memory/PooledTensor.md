---
title: "PooledTensor<T>"
description: "A RAII wrapper that automatically returns a pooled tensor to its pool when disposed."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Memory`

A RAII wrapper that automatically returns a pooled tensor to its pool when disposed.
This class ensures tensors are properly returned to the pool even if an exception occurs.

## How It Works

PooledTensor provides a safe way to use pooled tensors with the using statement,
ensuring the tensor is returned to the pool when it goes out of scope.

This is implemented as a class (not struct) to ensure reference semantics.
Multiple references to the same PooledTensor share disposal state, preventing
double-dispose issues that could corrupt the pool.

Example usage:

Alternative explicit cast usage:

**Thread Safety:** The Dispose method is idempotent - calling it multiple times
is safe and will only return the tensor to the pool once.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PooledTensor(TensorPool<>,Tensor<>)` | Initializes a new instance of the `PooledTensor` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsDisposed` | Gets whether this wrapper has been disposed and the tensor returned to the pool. |
| `Tensor` | Gets the underlying tensor managed by this wrapper. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Returns the tensor to the pool. |
| `op_Explicit(PooledTensor<>)~Tensor<>` | Explicitly converts a PooledTensor to its underlying Tensor. |

