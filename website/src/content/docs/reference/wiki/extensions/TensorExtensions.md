---
title: "TensorExtensions"
description: "TensorExtensions — Helpers & Utilities in AiDotNet.Extensions."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

_No summary documentation available yet._

## Methods

| Method | Summary |
|:-----|:--------|
| `ConcatenateTensors(Tensor<>,Tensor<>)` | Concatenates two tensors along the last dimension. |
| `ConvertToMatrix(Tensor<>)` | Converts a tensor to a matrix for various calculations. |
| `CreateOnesTensor(Int32)` | Creates a tensor initialized with all ones. |
| `CreateXavierInitializedTensor(Int32[],Double,Random)` | Creates and initializes a tensor with Xavier/Glorot initialization using random values. |
| `ForEachPosition(Tensor<>,Func<Int32[],,Boolean>)` | Iterates through all positions in the tensor and applies a function to each position. |
| `ForEachPositionRecursive(Tensor<>,Int32[],Int32,Func<Int32[],,Boolean>)` | Helper method for recursive iteration through tensor positions. |
| `HeStddev(Int32)` | Calculates the He standard deviation for weight initialization with ReLU activations. |
| `Unflatten(Tensor<>,Vector<>)` | Converts a flattened vector back into a tensor with the specified shape. |
| `XavierStddev(Int32,Int32)` | Calculates the Xavier/Glorot standard deviation for weight initialization. |

