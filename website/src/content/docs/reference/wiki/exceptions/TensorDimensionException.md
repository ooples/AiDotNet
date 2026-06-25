---
title: "TensorDimensionException"
description: "Exception thrown when a tensor's dimension doesn't match the expected value."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when a tensor's dimension doesn't match the expected value.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorDimensionException(Int32,Int32,Int32,String,String)` | Initializes a new instance of the `TensorDimensionException` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualValue` | The actual value of the dimension. |
| `Component` | The component where the dimension mismatch occurred. |
| `DimensionIndex` | The index of the dimension that doesn't match. |
| `ExpectedValue` | The expected value of the dimension. |
| `Operation` | The operation being performed when the mismatch was detected. |

