---
title: "InvalidDataValueException"
description: "Exception thrown when input data contains invalid values such as NaN or infinity."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when input data contains invalid values such as NaN or infinity.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InvalidDataValueException` | Creates a new instance of the InvalidDataValueException class. |
| `InvalidDataValueException(String)` | Creates a new instance of the InvalidDataValueException class with a specified error message. |
| `InvalidDataValueException(String,Exception)` | Creates a new instance of the InvalidDataValueException class with a specified error message and a reference to the inner exception that is the cause of this exception. |
| `InvalidDataValueException(String,String,String)` | Creates a new instance of the InvalidDataValueException class with a specified error message and context information about where the exception occurred. |
| `InvalidDataValueException(String,String,String,Exception)` | Creates a new instance of the InvalidDataValueException class with a specified error message, context information, and a reference to the inner exception. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Component` | The component where the invalid data was detected. |
| `Operation` | The operation being performed when the invalid data was detected. |

