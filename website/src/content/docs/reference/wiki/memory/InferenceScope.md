---
title: "InferenceScope<T>"
description: "Provides ambient (thread-local) context support for `InferenceContext`."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Memory`

Provides ambient (thread-local) context support for `InferenceContext`.
Allows code to access the current inference context without parameter threading.

## How It Works

InferenceScope enables the ambient context pattern, where a context is set once
and then accessible throughout the call stack without passing it as a parameter.

Usage example:

Scopes can be nested. When disposed, the previous scope is automatically restored:

## Properties

| Property | Summary |
|:-----|:--------|
| `Current` | Gets or sets the current inference context for this thread. |
| `IsActive` | Gets whether an inference scope is currently active on this thread. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Begin(InferenceContext<>)` | Begins a new inference scope with the specified context. |
| `RentOrCreate(Int32[])` | Rents a tensor from the current context if active, otherwise creates a new tensor. |
| `RentOrCreateLike(Tensor<>)` | Rents or creates a tensor with the same shape as the template. |

