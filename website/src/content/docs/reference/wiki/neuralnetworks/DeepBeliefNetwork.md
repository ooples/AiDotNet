---
title: "DeepBeliefNetwork<T>"
description: "Represents a Deep Belief Network, a generative graphical model composed of multiple layers of Restricted Boltzmann Machines."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Deep Belief Network, a generative graphical model composed of multiple layers of Restricted Boltzmann Machines.

## For Beginners

A Deep Belief Network is like a tower of pattern-recognizing layers.

Imagine building a tower where:

- Each floor of the tower is a Restricted Boltzmann Machine (RBM)
- The bottom floor learns simple patterns from the raw data
- Each higher floor learns more complex patterns based on what the floor below it discovered
- The tower is built and trained one floor at a time, from bottom to top

For example, if analyzing images of faces:

- The first floor might learn to detect edges and basic shapes
- The middle floors might recognize features like eyes, noses, and mouths
- The top floors might identify complete facial expressions or identities

This layer-by-layer approach helps the network discover meaningful patterns even when you don't have a lot of labeled examples.

## How It Works

A Deep Belief Network (DBN) is a probabilistic, generative model composed of multiple layers of stochastic 
latent variables. It is built by stacking multiple Restricted Boltzmann Machines (RBMs), where each RBM's 
hidden layer serves as the input layer for the next RBM. DBNs are trained using a two-phase approach: 
an unsupervised pre-training phase followed by a supervised fine-tuning phase. This allows them to learn 
complex patterns in data even with limited labeled examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepBeliefNetwork` | Initializes a new instance with default architecture settings. |
| `DeepBeliefNetwork(NeuralNetworkArchitecture<>,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,DeepBeliefNetworkOptions)` | Initializes a new instance of the `DeepBeliefNetwork` class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Indicates whether the network supports training (learning from data). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateLoss(Tensor<>,Tensor<>)` | Calculates the loss between predicted and expected outputs using the appropriate loss function. |
| `CalculateOutputGradients(Vector<>,Vector<>)` | Calculates the gradients of the loss with respect to the network outputs. |
| `CalculateReconstructionError(Tensor<>,Tensor<>)` | Calculates the reconstruction error between original and reconstructed data. |
| `DeepCopy` | Creates a new instance of the deep belief network model. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data for the Deep Belief Network. |
| `GetGradientShape` | Gets the shape of the gradient tensor for all layers in the Deep Belief Network. |
| `GetModelMetadata` | Gets metadata about the Deep Belief Network model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the Deep Belief Network based on the architecture. |
| `PreTrain(Tensor<>,Int32,Double)` | Greedy layer-wise pre-training of the stacked RBMs via Contrastive Divergence per Hinton 2006 ("A fast learning algorithm for deep belief nets") and Hinton & Salakhutdinov 2006 ("Reducing the Dimensionality of Data with Neural Networks"). |
| `PredictCore(Tensor<>)` | Makes a prediction using the current state of the Deep Belief Network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data for the Deep Belief Network. |
| `Train(Tensor<>,Tensor<>)` | Performs supervised fine-tuning of the Deep Belief Network after pre-training. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the Deep Belief Network. |
| `ValidateRbmLayers` | Validates that the RBM layers form a valid sequence for a Deep Belief Network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_batchSize` | Gets or sets the batch size for training. |
| `_epochs` | Gets or sets the number of epochs for fine-tuning. |
| `_learningRate` | Gets or sets the learning rate for parameter updates during fine-tuning. |
| `_lossFunction` | Gets or sets the loss function used for fine-tuning. |
| `_rbmLayers` | Gets or sets the list of RBM layers for greedy layer-wise pre-training. |

