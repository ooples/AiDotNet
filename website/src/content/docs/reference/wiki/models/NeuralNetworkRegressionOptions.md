---
title: "NeuralNetworkRegressionOptions<T, TInput, TOutput>"
description: "Configuration options for neural network regression models, providing fine-grained control over network architecture, training parameters, activation functions, and optimization strategies."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for neural network regression models, providing fine-grained control over
network architecture, training parameters, activation functions, and optimization strategies.

## For Beginners

Neural networks are AI models inspired by the human brain that can learn complex patterns.

Imagine building a system to predict house prices based on features like size, location, and age:

- Traditional methods might use simple formulas (like linear regression)
- Neural networks can discover complicated relationships that simple formulas miss

A neural network consists of layers of interconnected "neurons":

- Input layer: Receives your data (like house size, number of bedrooms)
- Hidden layers: Process the information and discover patterns
- Output layer: Produces the prediction (like the estimated house price)

The network "learns" by:

- Making predictions on training data
- Comparing those predictions to the actual values
- Adjusting its internal connections to reduce the errors
- Repeating this process many times

This class lets you configure every aspect of a neural network designed specifically for regression
(predicting continuous values like prices, temperatures, or scores), from its structure to how it learns.

## How It Works

Neural network regression is a powerful approach for modeling complex nonlinear relationships between
input features and continuous output variables. This class encapsulates the full range of parameters
needed to define and train a neural network for regression tasks. It allows for customization of
network depth and width, training duration and batch size, activation functions, loss functions, and
optimization algorithms. These options collectively determine the network's capacity, learning behavior,
and computational requirements, making them crucial for achieving optimal predictive performance while
managing training time and resource usage.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the number of training examples used in one iteration of model training. |
| `Epochs` | Gets or sets the number of complete passes through the training dataset during model training. |
| `HiddenActivationFunction` | Gets or sets the activation function applied to the outputs of hidden layer neurons. |
| `HiddenVectorActivation` | Gets or sets the vector activation function applied to the outputs of hidden layer neurons. |
| `LayerSizes` | Gets or sets the sizes of each layer in the neural network, including input, hidden, and output layers. |
| `LearningRate` | Gets or sets the step size used for updating model weights during gradient descent. |
| `LossFunction` | Gets or sets the function used to calculate the error between predicted and actual values. |
| `Optimizer` | Gets or sets the optimization algorithm used to update the network weights during training. |
| `OutputActivationFunction` | Gets or sets the activation function applied to the outputs of the final layer neurons. |
| `OutputVectorActivation` | Gets or sets the vector activation function applied to the outputs of the final layer neurons. |

