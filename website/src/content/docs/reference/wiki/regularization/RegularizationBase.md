---
title: "RegularizationBase<T, TInput, TOutput>"
description: "RegularizationBase<T, TInput, TOutput> — Base Classes in AiDotNet.Regularization."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Regularization`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RegularizationBase(RegularizationOptions)` | Initializes a new instance of the RegularizationBase class with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vector operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetOptions` | Gets the configuration options for this regularization technique. |
| `Regularize(,)` | Adjusts the gradient vector to account for regularization during optimization. |
| `Regularize(Matrix<>)` | Applies regularization to a matrix of input features. |
| `Regularize(Vector<>)` | Applies regularization to model coefficients. |
| `Regularize(Vector<>,Vector<>)` | Vector-direct gradient-aware regularization overload — same semantics as `Regularize(` but avoids the wrap/unwrap round-trip when both inputs are already flat vectors (the typical case inside a gradient-based optimizer's per-batch step). |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations appropriate for the generic type T. |
| `Options` | Configuration options for the regularization technique. |

