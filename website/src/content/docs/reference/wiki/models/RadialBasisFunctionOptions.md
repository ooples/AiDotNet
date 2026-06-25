---
title: "RadialBasisFunctionOptions"
description: "Configuration options for Radial Basis Function (RBF) models, a type of artificial neural network that uses radial basis functions as activation functions for approximating complex non-linear relationships."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Radial Basis Function (RBF) models, a type of artificial neural network
that uses radial basis functions as activation functions for approximating complex non-linear relationships.

## For Beginners

Radial Basis Function networks are a special kind of AI model that's good at finding patterns in data.

Think about weather prediction:

- Traditional models might try to find one formula that works for the whole world
- But an RBF network places "experts" at different locations
- Each "expert" (or center) specializes in predicting weather in their local area
- The final prediction combines opinions from nearby experts, with closer ones having more influence

What this technique does:

- It places a number of "centers" throughout your data
- Each center is like a spotlight that illuminates the nearby data points
- The model learns how strong each spotlight should be
- Predictions are made by seeing how much light falls on new data points

This is especially useful when:

- Your data has clusters or regions with different patterns
- You need a model that can adapt to different "neighborhoods" in your data
- You want smooth transitions between these different regions
- The relationship between inputs and outputs changes across your data space

For example, in image recognition, different RBF centers might specialize in detecting different
shapes or textures, and the combined output helps identify the complete image.

This class lets you configure how the RBF network is structured and initialized.

## How It Works

Radial Basis Function networks are a specialized type of neural network that utilize radially symmetric
functions (typically Gaussian) centered at specific points in the feature space. These networks excel at
function approximation, interpolation, and classification tasks. RBF networks consist of an input layer,
a hidden layer with RBF activation functions, and an output layer. Each neuron in the hidden layer represents
a radial basis function centered at a particular point. The output of the network is typically a linear
combination of these basis functions. RBF networks are known for their ability to model complex non-linear
relationships while often requiring less training time than traditional multilayer perceptrons. They are
particularly effective for problems where the data exhibits localized patterns or when smooth interpolation
between data points is desired.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfCenters` | Gets or sets the number of RBF centers (hidden neurons) in the network. |

