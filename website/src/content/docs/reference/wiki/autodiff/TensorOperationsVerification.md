---
title: "TensorOperationsVerification<T>"
description: "Verifies that TensorOperations autodiff gradients match numerical gradients."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Autodiff.Testing`

Verifies that TensorOperations autodiff gradients match numerical gradients.

## For Beginners

This class tests that our automatic differentiation is correct.

The process:

1. We have operations like ReLU, Sigmoid, Add, etc. in TensorOperations
2. Each operation computes gradients using autodiff (our fast implementation)
3. We also compute gradients numerically (slow but always correct)
4. If they match, our autodiff is correct!

This is essential for:

- Testing new operations before using them in training
- Debugging gradient issues in neural networks
- Ensuring mathematical correctness of backward passes

Example usage:

## How It Works

This class provides comprehensive verification of TensorOperations gradient implementations
by comparing autodiff results with numerically computed gradients using the central difference method.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorOperationsVerification` | Initializes with default configuration. |
| `TensorOperationsVerification(TensorOperationsVerification<>.VerificationConfig)` | Initializes with custom configuration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateOnes(Int32[])` | Creates a tensor filled with ones. |
| `CreateTestTensor(Int32[],Double,Double,Int32)` | Creates a test tensor with random values in the specified range. |
| `GetTopologicalOrder(ComputationNode<>)` | Gets topological order for gradient computation. |
| `RunBackward(ComputationNode<>)` | Runs the backward pass for a computation graph starting from the given node. |
| `VerifyAdd(Int32[])` | Verifies Add operation gradients. |
| `VerifyAllOperations` | Runs verification for all standard operations. |
| `VerifyBinaryOperation(Func<ComputationNode<>,ComputationNode<>,ComputationNode<>>,Tensor<>,Tensor<>,String)` | Verifies a binary operation's gradient computation. |
| `VerifyElementwiseDivide(Int32[])` | Verifies ElementwiseDivide operation gradients. |
| `VerifyElementwiseMultiply(Int32[])` | Verifies ElementwiseMultiply operation gradients. |
| `VerifyExp(Int32[])` | Verifies Exp operation gradients. |
| `VerifyLeakyReLU(Int32[],Double)` | Verifies LeakyReLU operation gradients. |
| `VerifyLog(Int32[])` | Verifies Log operation gradients. |
| `VerifyNegate(Int32[])` | Verifies Negate operation gradients. |
| `VerifyReLU(Int32[])` | Verifies ReLU operation gradients. |
| `VerifySigmoid(Int32[])` | Verifies Sigmoid operation gradients. |
| `VerifySqrt(Int32[])` | Verifies Sqrt operation gradients. |
| `VerifySquare(Int32[])` | Verifies Square operation gradients. |
| `VerifySubtract(Int32[])` | Verifies Subtract operation gradients. |
| `VerifyTanh(Int32[])` | Verifies Tanh operation gradients. |
| `VerifyUnaryOperation(Func<ComputationNode<>,ComputationNode<>>,Tensor<>,String)` | Verifies a unary operation's gradient computation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | The numeric operations appropriate for the generic type T. |

