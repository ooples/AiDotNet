---
title: "MultilayerPerceptronOptions<T, TInput, TOutput>"
description: "Configuration options for Multilayer Perceptron (MLP), a type of feedforward artificial neural network that consists of multiple layers of neurons."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Multilayer Perceptron (MLP), a type of feedforward artificial neural
network that consists of multiple layers of neurons.

## For Beginners

A Multilayer Perceptron (MLP) is a basic type of neural network that
can learn to recognize patterns and make predictions from data.

Think of an MLP like a system of interconnected filters that work together:

- The input layer receives your data (like the temperature, humidity, and pressure for weather prediction)
- The hidden layers process this information through a series of transformations
- The output layer provides the prediction (like "chance of rain: 70%")

As the network trains, it gradually adjusts thousands of internal settings (weights) to get better
at making accurate predictions. This process is similar to how a child learns to recognize animals:
at first they make many mistakes, but with each example, they get better at identifying the patterns
that distinguish a cat from a dog.

This class lets you configure every aspect of your neural network: how many layers it has, how it learns,
how quickly it adapts, and much more. The default settings provide a good starting point, but you may
need to adjust them based on your specific problem.

## How It Works

The Multilayer Perceptron is a versatile neural network architecture capable of learning complex
non-linear relationships between inputs and outputs. It consists of an input layer, one or more hidden
layers, and an output layer. Each neuron in a layer is connected to all neurons in the next layer,
forming a fully connected network. The MLP learns through a process called backpropagation, where the
network parameters are adjusted to minimize a loss function using gradient-based optimization techniques.
This class provides comprehensive configuration options for the network architecture, training process,
activation functions, and optimization strategy.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the number of training examples used in each parameter update step. |
| `HiddenActivation` | Gets or sets the activation function used in the hidden layers of the network. |
| `HiddenVectorActivation` | Gets or sets the vector-based activation function used in the hidden layers of the network. |
| `LayerSizes` | Gets or sets the sizes of each layer in the neural network, including input, hidden, and output layers. |
| `LearningRate` | Gets or sets the learning rate that controls the step size in each update of the model parameters. |
| `LossFunction` | Gets or sets the loss function used to calculate the error between predictions and targets. |
| `MaxEpochs` | Gets or sets the maximum number of complete passes through the training dataset. |
| `OutputActivation` | Gets or sets the activation function used in the output layer of the network. |
| `OutputVectorActivation` | Gets or sets the vector-based activation function used in the output layer of the network. |
| `Verbose` | Gets or sets whether to display detailed progress information during training. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_optimizer` | Gets or sets the optimization algorithm used to update the network weights during training. |

