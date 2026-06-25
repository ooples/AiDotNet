---
title: "InvalidInputDimensionException"
description: "Exception thrown when input data dimensions are invalid for a specific algorithm or operation."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when input data dimensions are invalid for a specific algorithm or operation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InvalidInputDimensionException` | Creates a new instance of the InvalidInputDimensionException class. |
| `InvalidInputDimensionException(String)` | Creates a new instance of the InvalidInputDimensionException class with a specified error message. |
| `InvalidInputDimensionException(String,Exception)` | Creates a new instance of the InvalidInputDimensionException class with a specified error message and a reference to the inner exception that is the cause of this exception. |
| `InvalidInputDimensionException(String,String,String)` | Creates a new instance of the InvalidInputDimensionException class with a specified error message and context information about where the exception occurred. |
| `InvalidInputDimensionException(String,String,String,Exception)` | Creates a new instance of the InvalidInputDimensionException class with a specified error message, context information, and a reference to the inner exception. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Component` | The component where the dimension mismatch occurred. |
| `Operation` | The operation being performed when the mismatch was detected. |

