---
title: "INeuralNetwork<T>"
description: "Defines the core functionality for neural network models in the AiDotNet library."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the core functionality for neural network models in the AiDotNet library.

## How It Works

This interface provides methods for making predictions, updating model parameters,
saving and loading models, and controlling training behavior.

**For Beginners:** A neural network is a type of machine learning model inspired by the human brain.

Think of a neural network as a system that learns patterns:

- It's made up of interconnected "neurons" (small computing units)
- These neurons are organized in layers (input layer, hidden layers, output layer)
- Each connection between neurons has a "weight" (importance)
- The network learns by adjusting these weights based on examples it sees

For example, in an image recognition neural network:

- The input layer receives pixel values from an image
- Hidden layers detect patterns like edges, shapes, and textures
- The output layer determines what the image contains (e.g., "cat" or "dog")

Neural networks are powerful because they can:

- Learn complex patterns from data
- Make predictions on new, unseen data
- Improve their accuracy with more training

This interface provides the essential methods needed to work with neural networks in AiDotNet.

## Methods

| Method | Summary |
|:-----|:--------|
| `ForwardWithMemory(Tensor<>)` | Performs a forward pass while storing intermediate activations for backpropagation. |
| `GetLastLoss` | Gets the loss value from the most recent training step. |
| `GetParameterGradients` | Gets the gradients computed during the most recent backpropagation. |
| `SetTrainingMode(Boolean)` | Sets whether the neural network is in training mode or inference (prediction) mode. |
| `UpdateParameters(Vector<>)` | Updates the internal parameters (weights and biases) of the neural network. |

