---
title: "GDAS<T>"
description: "Gradient-based Differentiable Architecture Search with Gumbel-Softmax sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

Gradient-based Differentiable Architecture Search with Gumbel-Softmax sampling.
GDAS uses Gumbel-Softmax to make the architecture search fully differentiable
while maintaining discrete selection during forward pass.

Reference: "Searching for A Robust Neural Architecture in Four GPU Hours" (CVPR 2019)

## For Beginners

GDAS finds good neural network architectures in just
4 GPU hours using Gumbel-Softmax sampling. This technique makes discrete architecture
choices (which layer type to use) differentiable, so gradient descent can optimize
them directly. Think of it as using calculus to navigate a menu of design choices
rather than trying them all one by one.

## Methods

| Method | Summary |
|:-----|:--------|
| `AnnealTemperature(Int32,Int32)` | Anneals the Gumbel-Softmax temperature during training |
| `DeriveArchitecture` | Derives the discrete architecture by selecting the operation with highest weight |
| `GetArchitectureGradients` | Gets architecture gradients |
| `GetArchitectureParameters` | Gets architecture parameters for optimization |
| `GetTemperature` | Gets current temperature |
| `GumbelSoftmax(Matrix<>,Boolean)` | Applies Gumbel-Softmax sampling to architecture parameters. |

