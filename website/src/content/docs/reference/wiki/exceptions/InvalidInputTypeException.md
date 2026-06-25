---
title: "InvalidInputTypeException"
description: "Exception thrown when a neural network receives an input type that doesn't match its requirements."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Exceptions`

Exception thrown when a neural network receives an input type that doesn't match its requirements.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InvalidInputTypeException(InputType,InputType,String)` | Initializes a new instance of the `InvalidInputTypeException` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActualInputType` | The actual input type provided. |
| `ExpectedInputType` | The expected input type for the neural network. |
| `NetworkType` | The type of neural network that requires the specific input type. |

