---
title: "SphericalSoftmaxActivation<T>"
description: "Implements the Spherical Softmax activation function, which normalizes inputs to the unit sphere before applying softmax."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Spherical Softmax activation function, which normalizes inputs to the unit sphere before applying softmax.

## For Beginners

Spherical Softmax is a special version of the Softmax function that's often used in the
output layer of neural networks for classification tasks. 

The standard Softmax converts a vector of numbers into probabilities that sum to 1, but it can sometimes
have numerical issues with very large or very small numbers. Spherical Softmax adds an extra step by first
"normalizing" the input vector (making its length equal to 1) before applying the regular Softmax steps.

Think of it like this:

1. First, we adjust all the input values so that they form a point on a sphere with radius 1
2. Then we apply the regular Softmax calculation to these adjusted values

This approach can make the neural network training more stable and sometimes leads to better performance,
especially when dealing with high-dimensional data or when the input values can vary widely in magnitude.

## How It Works

Spherical Softmax is a variation of the standard Softmax function that first normalizes the input vector
to have unit length (projects it onto the unit sphere) before applying the exponential and normalization steps.
This can help improve numerical stability and performance in certain neural network architectures.

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate(Tensor<>)` | Applies SphericalSoftmax to a tensor via engine primitives so the gradient tape records every step. |
| `Activate(Vector<>)` | Applies the Spherical Softmax activation function to a vector of inputs. |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative(Vector<>)` | Calculates the Jacobian matrix of the Spherical Softmax function for a given input vector. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NormalizationEpsilon` | Numerical-stability epsilon added to the squared-norm before the sqrt to avoid division-by-zero when the input vector is exactly zero. |

