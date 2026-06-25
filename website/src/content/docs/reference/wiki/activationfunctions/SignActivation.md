---
title: "SignActivation<T>"
description: "Implements the Sign activation function, which returns -1 for negative inputs, 1 for positive inputs, and 0 for zero."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Sign activation function, which returns -1 for negative inputs, 1 for positive inputs, and 0 for zero.

## For Beginners

The Sign function is like a simple categorizer that looks at a number and tells you if it's 
negative (-1), zero (0), or positive (1). It's one of the simplest activation functions and is useful when 
you want your neural network to make clear-cut decisions rather than produce probabilities or continuous values.
However, because it has sharp "jumps" in its output and its derivative is zero almost everywhere, it's rarely 
used in modern neural networks that rely on gradient-based learning.

## How It Works

The Sign function is a simple non-linear activation function that categorizes inputs into three distinct outputs:
-1 (negative), 0 (zero), or 1 (positive). Unlike smooth activation functions, Sign has sharp transitions.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the Sign activation function to a single input value. |
| `Activate(Tensor<>)` | Applies the Sign activation function to each element in a tensor. |
| `Activate(Vector<>)` | Applies the Sign activation function to each element in a vector. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the Sign function for a single input value. |
| `Derivative(Tensor<>)` | Calculates the derivative of the Sign function for a tensor input. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the Sign function for a vector input. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

