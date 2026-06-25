---
title: "INeuralNetworkModel<T>"
description: "Defines the contract for neural network models with advanced architectural introspection capabilities."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for neural network models with advanced architectural introspection capabilities.

## How It Works

This interface extends the basic neural network functionality with methods for accessing
the internal architecture and layer-wise activations of neural networks.

**For Beginners:** This interface represents a neural network that can tell you about its structure.

Think of INeuralNetworkModel as a neural network with "x-ray vision" into its own structure:

- It can show you what each layer produces (activations)
- It can describe its own architecture (how it's built)
- It combines all the basic neural network abilities with introspection capabilities

For example, if you're debugging or analyzing a neural network:

- You can see what each layer outputs for a given input
- You can examine the network's structure (number of layers, layer types, connections)
- You can understand how information flows through the network

This is particularly useful for:

- Debugging neural networks (seeing where things go wrong)
- Understanding what the network has learned
- Visualizing how the network processes information
- Implementing advanced techniques like transfer learning or feature extraction

This interface is typically implemented by neural network base classes that provide
comprehensive access to their internal structure and computations.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetArchitecture` | Gets the architectural structure of the neural network. |
| `GetNamedLayerActivations(Tensor<>)` | Gets the intermediate activations from each layer when processing the given input with named keys. |

