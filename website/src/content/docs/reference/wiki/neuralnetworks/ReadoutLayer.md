---
title: "ReadoutLayer<T>"
description: "Represents a readout layer that performs the final mapping from features to output in a neural network."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a readout layer that performs the final mapping from features to output in a neural network.

## For Beginners

This layer serves as the final "decision maker" in a neural network.

Think of the ReadoutLayer as a panel of judges in a competition:

- Each judge (output neuron) receives information from all contestants (input features)
- Each judge has their own preferences (weights) for different skills
- Judges combine all this information to make their final scores (outputs)
- An activation function then shapes these scores into the desired format

For example, in an image classification network:

- Previous layers extract features like edges, shapes, and patterns
- The ReadoutLayer takes all these features and combines them into class scores
- If there are 10 possible classes, the ReadoutLayer might have 10 outputs
- Each output represents the network's confidence that the image belongs to that class

This layer learns which features are most important for each output category during training.

## How It Works

The ReadoutLayer is typically used as the final layer in a neural network to transform features 
extracted by previous layers into the desired output format. It applies a linear transformation 
(weights and bias) followed by an activation function. This layer is similar to a dense or fully 
connected layer but is specifically designed for outputting the final results.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReadoutLayer(Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `ReadoutLayer` class with a scalar activation function. |
| `ReadoutLayer(Int32,Int32,IVectorActivationFunction<>,IInitializationStrategy<>)` | Initializes a new instance of the `ReadoutLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU execution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass of the readout layer. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass on GPU using FusedLinearGpu. |
| `GetParameterGradients` | Sets the trainable parameters of the readout layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters of the readout layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters(Int32,Int32)` | Initializes the weights and biases of the readout layer with small random values and zeros. |
| `ResetState` | Resets the internal state of the readout layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates the parameters of the readout layer using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bias` | Tensor storing the bias parameters for each output neuron. |
| `_biasGradients` | Tensor storing the gradients of the loss with respect to the bias parameters. |
| `_lastInput` | Stores the input tensor from the most recent forward pass for use in backpropagation. |
| `_lastOutput` | Stores the output tensor (post-activation) from the most recent forward pass for use in backpropagation. |
| `_lastPreActivation` | Stores the pre-activation output tensor from the most recent forward pass. |
| `_weightGradients` | Tensor storing the gradients of the loss with respect to the weight parameters. |
| `_weights` | Tensor storing the weight parameters for connections between inputs and outputs. |

