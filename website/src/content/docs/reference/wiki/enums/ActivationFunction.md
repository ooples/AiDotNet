---
title: "ActivationFunction"
description: "Represents different activation functions used in neural networks and deep learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different activation functions used in neural networks and deep learning.

## For Beginners

Activation functions are mathematical operations that determine whether a neuron in a 
neural network should be "activated" (output a signal) or not.

Think of a neuron as a decision-maker that:

1. Receives multiple inputs
2. Calculates a weighted sum of these inputs
3. Applies an activation function to decide what value to output

Without activation functions, neural networks would just be linear models (like basic regression), 
unable to learn complex patterns. Activation functions add non-linearity, allowing networks to learn 
complicated relationships in data.

Different activation functions have different properties that make them suitable for different tasks. 
Choosing the right activation function can significantly impact how well your neural network learns 
and performs.

## Fields

| Field | Summary |
|:-----|:--------|
| `ELU` | Exponential Linear Unit - smooth version of ReLU that can output negative values. |
| `GELU` | Gaussian Error Linear Unit - a smooth activation function that performs well in transformers and language models. |
| `Identity` | Identity function - returns the input value unchanged, providing a direct pass-through. |
| `LeakyReLU` | Leaky Rectified Linear Unit - similar to ReLU but allows a small gradient for negative inputs. |
| `LiSHT` | Linearly Scaled Hyperbolic Tangent - a self-regularized activation function. |
| `Linear` | Linear activation - simply returns the input value unchanged. |
| `ReLU` | Rectified Linear Unit - returns 0 for negative inputs and the input value for positive inputs. |
| `SELU` | Scaled Exponential Linear Unit - self-normalizing version of ELU. |
| `Sigmoid` | Sigmoid function - maps any input to a value between 0 and 1. |
| `SoftSign` | SoftSign function - maps inputs to values between -1 and 1 with a smoother approach to the asymptotes. |
| `Softmax` | Softmax function - converts a vector of values to a probability distribution. |
| `Softplus` | Softplus function - a smooth approximation of the ReLU function. |
| `Swish` | Swish function - a self-gated activation function developed by researchers at Google. |
| `Tanh` | Hyperbolic Tangent - maps any input to a value between -1 and 1. |

