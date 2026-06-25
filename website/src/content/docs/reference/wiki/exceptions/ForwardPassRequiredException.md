---
title: "ForwardPassRequiredException"
description: "Exception thrown when an operation is attempted before a required forward pass has been completed."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when an operation is attempted before a required forward pass has been completed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ForwardPassRequiredException(String,String)` | Initializes a new instance of the `ForwardPassRequiredException` class for a layer. |
| `ForwardPassRequiredException(String,String,String)` | Initializes a new instance of the `ForwardPassRequiredException` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ComponentName` | The name of the component where the exception occurred. |
| `ComponentType` | The type of the component where the exception occurred. |
| `Operation` | The operation that was attempted. |

