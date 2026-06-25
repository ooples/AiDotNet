---
title: "PrototypeVector<T>"
description: "Prototype vector class that delegates operations to the execution engine."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Prototypes`

Prototype vector class that delegates operations to the execution engine.
Used to validate the Execution Engine pattern before production integration.

## How It Works

This is a PROTOTYPE for Phase A validation. The production version will integrate
directly with the existing Vector<T> class.

PrototypeVector demonstrates:

- Engine delegation pattern
- Vectorized operations (no element-wise for-loops)
- Transparent GPU acceleration
- Zero constraint cascade

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrototypeVector(Int32)` | Initializes a new instance of the PrototypeVector class with the specified length. |
| `PrototypeVector(Vector<>)` | Initializes a new instance of the PrototypeVector class from an existing Vector. |
| `PrototypeVector([])` | Initializes a new instance of the PrototypeVector class from an array. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Item(Int32)` | Gets or sets the element at the specified index. |
| `Length` | Gets the length of the vector. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(PrototypeVector<>)` | Adds two vectors element-wise using the current execution engine. |
| `Divide()` | Divides the vector by a scalar. |
| `Divide(PrototypeVector<>)` | Divides this vector by another vector element-wise. |
| `FromArray([])` | Creates a vector from an array. |
| `Multiply()` | Multiplies the vector by a scalar. |
| `Multiply(PrototypeVector<>)` | Multiplies two vectors element-wise (Hadamard product). |
| `Ones(Int32)` | Creates a vector filled with ones. |
| `Power()` | Raises each element to the specified power. |
| `Sqrt` | Computes the square root of each element. |
| `Subtract(PrototypeVector<>)` | Subtracts another vector from this vector element-wise. |
| `ToArray` | Converts to array. |
| `ToString` | Returns a string representation of the vector. |
| `ToVector` | Gets the underlying Vector<T> data. |
| `Zeros(Int32)` | Creates a vector filled with zeros. |

