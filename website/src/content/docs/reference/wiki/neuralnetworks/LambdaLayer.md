---
title: "LambdaLayer<T>"
description: "Represents a customizable layer that applies user-defined functions for both forward and backward passes."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a customizable layer that applies user-defined functions for both forward and backward passes.

## For Beginners

This layer lets you create your own custom operations in a neural network.

Think of the Lambda Layer as a "do-it-yourself" layer where:

- You provide your own custom function to process the data
- You can optionally provide a custom function for the learning process
- It gives you flexibility to implement operations not covered by standard layers

For example, if you wanted to apply a special mathematical transformation that isn't
available in standard layers, you could define that transformation and use it in a Lambda Layer.

This is an advanced feature that gives you complete control when standard layers
don't provide what you need.

## How It Works

The Lambda Layer allows for custom transformations to be incorporated into a neural network by accepting
user-defined functions for both the forward and backward passes. This provides flexibility to implement
custom operations that aren't available as standard layers. The layer can optionally apply an activation
function after the custom transformation.

**JIT Compilation Support:** To enable JIT compilation, use the constructor that accepts
a traceable expression function (Func<ComputationNode<T>, ComputationNode<T>>) instead of
an opaque tensor function. The traceable function uses TensorOperations which can be compiled.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LambdaLayer(Int32[],Int32[],Func<ComputationNode<>,ComputationNode<>>,IActivationFunction<>)` | Initializes a new instance of the `LambdaLayer` class with a traceable expression for JIT compilation support. |
| `LambdaLayer(Int32[],Int32[],Func<Tensor<>,Tensor<>>,Func<Tensor<>,Tensor<>,Tensor<>>,IActivationFunction<>)` | Initializes a new instance of the `LambdaLayer` class with the specified shapes, functions, and element-wise activation function. |
| `LambdaLayer(Int32[],Int32[],Func<Tensor<>,Tensor<>>,Func<Tensor<>,Tensor<>,Tensor<>>,IVectorActivationFunction<>)` | Initializes a new instance of the `LambdaLayer` class with the specified shapes, functions, and vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the lambda layer. |
| `GetParameters` | Returns an empty vector since the lambda layer typically has no trainable parameters. |
| `ResetState` | Resets the internal state of the layer. |
| `UpdateParameters()` | Update parameters is a no-op for the lambda layer since it typically doesn't have trainable parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_backwardFunction` | The optional user-provided function that defines the backward pass transformation. |
| `_forwardFunction` | The user-provided function that defines the forward pass transformation. |
| `_lastInput` | Stores the input tensor from the last forward pass for use in the backward pass. |
| `_lastOutput` | Stores the output tensor from the last forward pass for use in the backward pass. |
| `_traceableExpression` | The optional traceable expression function for JIT compilation support. |

