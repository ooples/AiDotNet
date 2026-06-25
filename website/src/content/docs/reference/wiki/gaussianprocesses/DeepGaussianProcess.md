---
title: "DeepGaussianProcess<T>"
description: "Implements a Deep Gaussian Process (DGP) with multiple stacked GP layers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.GaussianProcesses`

Implements a Deep Gaussian Process (DGP) with multiple stacked GP layers.

## For Beginners

A Deep Gaussian Process stacks multiple GP layers on top of each other,
similar to how deep neural networks stack layers. This allows the model to learn hierarchical
representations and capture more complex patterns than a single GP.

How it works:

1. Input goes into the first GP layer
2. Output of layer 1 becomes input to layer 2
3. Continue through all layers
4. Final layer produces the prediction

Each layer can transform the data in different ways, allowing the model to learn
progressively more abstract representations.

## How It Works

Why use Deep GPs?

1. **Complex patterns**: Can model highly non-linear relationships that single GPs struggle with

2. **Hierarchical features**: Learn abstract representations at different levels

3. **Uncertainty propagation**: Unlike deep neural networks, DGPs propagate uncertainty

through all layers, giving more reliable confidence estimates

4. **Flexible architecture**: Can use different kernels at each layer

Limitations:

- More computationally expensive than single GPs
- Harder to train (requires careful initialization and optimization)
- May overfit on small datasets

**Implementation Note:** This is an experimental/research implementation with simplified
layer optimization. The current training uses a greedy layer-by-layer approach rather than
full ELBO gradient optimization. For production use cases requiring state-of-the-art DGP
performance, consider using specialized DGP libraries or extending this implementation
with proper doubly-stochastic variational inference.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepGaussianProcess(IKernelFunction<>,Int32,Int32,Int32,MatrixDecompositionType)` | Initializes a Deep Gaussian Process with a simple architecture. |
| `DeepGaussianProcess(IKernelFunction<>[],Int32[],Int32,Double,Int32,Int32,MatrixDecompositionType)` | Initializes a new Deep Gaussian Process with specified layer configurations. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumLayers` | Gets the number of layers in the deep GP. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fit(Matrix<>,Vector<>)` | Trains the Deep GP using variational inference. |
| `OptimizeLayers` | Optimizes all layer parameters using gradient-based optimization. |
| `Predict(Vector<>)` |  |
| `UpdateKernel(IKernelFunction<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_X` | The training input data. |
| `_decompositionType` | The matrix decomposition method to use. |
| `_layers` | The GP layers in the deep architecture. |
| `_learningRate` | Learning rate for optimization. |
| `_maxIterations` | Maximum optimization iterations. |
| `_numInducingPoints` | Number of inducing points per layer. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_numSamples` | Number of samples for Monte Carlo estimation. |
| `_y` | The training target values. |

