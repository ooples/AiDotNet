---
title: "TensorValidator"
description: "Provides validation methods for tensors and neural network operations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Validation`

Provides validation methods for tensors and neural network operations.

## Methods

| Method | Summary |
|:-----|:--------|
| `ValidateForwardPassPerformed(Tensor<>,String,String,String)` | Validates that a forward pass has been performed before a dependent operation. |
| `ValidateForwardPassPerformedForLayer(Tensor<>,String,String)` | Validates that a forward pass has been performed before a backward pass in a layer. |
| `ValidateRank(Tensor<>,Int32,String,String)` | Validates that a tensor has the expected rank (number of dimensions). |
| `ValidateShape(Tensor<>,Int32[],String,String)` | Validates that a tensor's shape matches the expected shape. |
| `ValidateShapesMatch(Tensor<>,Tensor<>,String,String)` | Validates that two tensors have the same shape. |

