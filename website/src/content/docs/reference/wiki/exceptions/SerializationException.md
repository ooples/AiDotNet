---
title: "SerializationException"
description: "Exception thrown when serialization or deserialization operations fail."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when serialization or deserialization operations fail.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SerializationException` | Creates a new instance of the SerializationException class. |
| `SerializationException(String)` | Creates a new instance of the SerializationException class with a specified error message. |
| `SerializationException(String,Exception)` | Creates a new instance of the SerializationException class with a specified error message and a reference to the inner exception that is the cause of this exception. |
| `SerializationException(String,String,String)` | Creates a new instance of the SerializationException class with a specified error message and context information about where the exception occurred. |
| `SerializationException(String,String,String,Exception)` | Creates a new instance of the SerializationException class with a specified error message, context information, and a reference to the inner exception. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Component` | The component where the serialization error occurred. |
| `Operation` | The operation being performed when the error was detected. |

