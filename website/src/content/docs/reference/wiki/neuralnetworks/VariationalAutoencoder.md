---
title: "VariationalAutoencoder<T>"
description: "Represents a Variational Autoencoder (VAE) neural network architecture, which is used for  generating new data similar to the training data and learning compressed representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Variational Autoencoder (VAE) neural network architecture, which is used for 
generating new data similar to the training data and learning compressed representations.

## For Beginners

A Variational Autoencoder is like a creative compression system.

Imagine you have a folder full of photos of cats:

- The encoder is like a person who studies all these photos and learns to describe any cat using just a few key attributes (like fur color, ear shape, size)
- These few attributes are the "latent space" - a much smaller representation of the data
- The special thing about a VAE is that instead of exact values, it describes each attribute as a range of possible values (a probability distribution)
- The decoder is like an artist who can take these attribute descriptions and draw a new cat based on them

This ability to work with probability distributions means:

- You can generate new, never-before-seen cats by sampling from these distributions
- The generated cats will look realistic because they follow the patterns learned from real cats
- You can smoothly transition between different types of cats by moving through the latent space

VAEs are used for image generation, data compression, anomaly detection, and other creative applications.

## How It Works

A Variational Autoencoder is a type of generative model that learns to encode input data into a 
probabilistic latent space and then decode samples from that space back into the original data space.
Unlike standard autoencoders, VAEs ensure the latent space has good properties for generating new samples 
by learning a distribution rather than a fixed encoding.

VAEs consist of:

- An encoder network that maps input data to a probability distribution in latent space
- A sampling mechanism that draws samples from this distribution
- A decoder network that maps samples from latent space back to the original data space

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VariationalAutoencoder` | Initializes a new instance of the `VariationalAutoencoder` class with the specified architecture, latent space size, and optional optimizer and loss function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight (beta parameter) for the KL divergence auxiliary loss. |
| `LatentSize` | Gets the size of the latent space dimension in the Variational Autoencoder. |
| `UseAuxiliaryLoss` | Gets or sets whether to use auxiliary loss (KL divergence) during training. |
| `_logVarianceLayer` | Gets or sets the layer that computes the log variance parameters of the latent distribution. |
| `_meanLayer` | Gets or sets the layer that computes the mean parameters of the latent distribution. |
| `_optimizer` | Gets or sets the gradient optimizer used for training the VAE. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateKLDivergence(Vector<>,Vector<>)` | Calculates the KL divergence between the learned distribution and a standard normal distribution. |
| `CalculateLatentGradients(Tensor<>)` | Calculates the gradients for the latent space (mean and log variance) of the VAE. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for the VAE, which is the KL divergence between the learned latent distribution and a standard normal distribution. |
| `CreateNewInstance` | Creates a new instance of the Variational Autoencoder with the same architecture and configuration. |
| `Decode(Vector<>)` | Decodes a vector from the latent space back to the original data space. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data for the Variational Autoencoder. |
| `Encode(Vector<>)` | Encodes an input vector into mean and log variance parameters in the latent space. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary loss computation. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Gets metadata about the Variational Autoencoder model. |
| `GetOptions` |  |
| `InitializeLayers` | Sets up the layers of the Variational Autoencoder based on the provided architecture. |
| `PredictCore(Tensor<>)` | Makes a prediction using the Variational Autoencoder by encoding the input, sampling from the latent space, and decoding. |
| `Reparameterize(Vector<>,Vector<>)` | Implements the reparameterization trick to sample from the latent distribution in a way that allows gradient flow. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data for the Variational Autoencoder. |
| `SetSpecificLayers` | Sets up references to the specific layers needed for the VAE's functionality. |
| `Train(Tensor<>,Tensor<>)` | Trains the Variational Autoencoder using the provided input data. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the VAE network. |
| `ValidateCustomLayers(List<ILayer<>>)` | Ensures that custom layers provided for the VAE meet the minimum requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastKLDivergence` | Stores the last computed KL divergence value for diagnostics. |
| `_lastLogVariance` | Stores the last computed log variance vector from the encoder for auxiliary loss computation. |
| `_lastMean` | Stores the last computed mean vector from the encoder for auxiliary loss computation. |

