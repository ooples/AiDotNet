---
title: "ConversionsHelper"
description: "Provides utility methods for converting between different data structures used in machine learning models."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides utility methods for converting between different data structures used in machine learning models.

## For Beginners

This helper class contains methods to convert between different mathematical 
objects used in AI (Tensor, Matrix, Vector). Think of it like a universal adapter that lets 
different parts of your AI system work together even when they expect different data formats.

## How It Works

For example, if one algorithm outputs a Matrix but another needs a Tensor, these methods
help you convert between them without writing complex conversion code yourself.

## Methods

| Method | Summary |
|:-----|:--------|
| `ConvertFitFunction(Func<,>)` | Converts a fit function that works with generic types to one that works with Matrix and Vector. |
| `ConvertObjectToVector(Object)` | Converts a generic object to a Vector. |
| `ConvertToMatrix()` | Converts an input of generic type to a Matrix. |
| `ConvertToScalar()` | Converts an output value to a scalar value of type T. |
| `ConvertToTensor(Object)` | Converts a Matrix or Vector to a Tensor. |
| `ConvertToVector()` | Converts an output of generic type to a Vector. |
| `ConvertVectorToInput(Vector<>,)` | Converts a Vector to the generic TInput type (Vector, Matrix, or Tensor) using a reference input for shape information. |
| `ConvertVectorToInputWithoutReference(Vector<>)` | Converts a Vector to the generic TInput type (Vector, Tensor, Matrix, T[], or scalar T) without requiring a reference input. |
| `GetSampleCount()` | Gets the sample count from supported input/output data types. |
| `MatrixToTensor(Matrix<>,Int32[])` | Converts a matrix to a tensor with the specified shape. |
| `TensorToMatrix(Tensor<>,Int32,Int32)` | Converts a tensor to a matrix with the specified dimensions. |
| `VectorToTensor(Vector<>,Int32[])` | Converts a vector to a tensor with the specified shape. |

