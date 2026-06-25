---
title: "TensorShapeMismatchException"
description: "Exception thrown when a tensor's shape doesn't match the expected shape."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when a tensor's shape doesn't match the expected shape.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorShapeMismatchException` | Initializes a new instance of the `TensorShapeMismatchException` class. |
| `TensorShapeMismatchException(Int32[],Int32[],String)` | Initializes a new instance of the `TensorShapeMismatchException` class with a simplified context. |
| `TensorShapeMismatchException(Int32[],Int32[],String,String)` | Initializes a new instance of the `TensorShapeMismatchException` class. |
| `TensorShapeMismatchException(Int32[],Int32[],String,String,Exception)` | Initializes a new instance of the `TensorShapeMismatchException` class with shape information and a reference to the inner exception that is the cause of this exception. |
| `TensorShapeMismatchException(String)` | Initializes a new instance of the `TensorShapeMismatchException` class with a specified error message. |
| `TensorShapeMismatchException(String,Exception)` | Initializes a new instance of the `TensorShapeMismatchException` class with a specified error message and a reference to the inner exception that is the cause of this exception. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualShape` | The actual shape of the tensor. |
| `Component` | The component where the shape mismatch occurred. |
| `ExpectedShape` | The expected shape of the tensor. |
| `Operation` | The operation being performed when the mismatch was detected. |

