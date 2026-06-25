---
title: "VectorLengthMismatchException"
description: "Exception thrown when a vector's length doesn't match the expected value."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when a vector's length doesn't match the expected value.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VectorLengthMismatchException(Int32,Int32,String,String)` | Initializes a new instance of the `VectorLengthMismatchException` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualLength` | The actual length of the vector. |
| `Component` | The component where the length mismatch occurred. |
| `ExpectedLength` | The expected length of the vector. |
| `Operation` | The operation being performed when the mismatch was detected. |

