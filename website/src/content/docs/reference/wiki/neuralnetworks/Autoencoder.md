---
title: "Autoencoder<T>"
description: "Represents an autoencoder neural network that can compress data into a lower-dimensional representation and reconstruct it."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents an autoencoder neural network that can compress data into a lower-dimensional representation and reconstruct it.

## For Beginners

An autoencoder is like a sophisticated compression and decompression system.

Think of it like this:

- The encoder part takes your original data (like an image) and compresses it into a smaller representation
- The middle layer (latent space) holds this compressed version of your data
- The decoder part takes this compressed version and tries to recreate the original data

For example, with images:

- You might compress a 256x256 pixel image (65,536 values) into just 100 numbers
- The network learns which features are most important to preserve
- It then learns to reconstruct the image from only those 100 numbers

This is useful for:

- Data compression
- Noise reduction (by removing noise during reconstruction)
- Feature learning (the compressed representation often contains meaningful features)
- Anomaly detection (unusual data is reconstructed poorly)

## How It Works

An autoencoder is a type of neural network designed to learn efficient data encodings in an unsupervised manner.
It consists of an encoder that compresses the input data into a lower-dimensional representation (the latent space)
and a decoder that reconstructs the original data from this representation. Autoencoders are trained to minimize
the difference between the original input and the reconstructed output.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Autoencoder` | Initializes a new instance of the `Autoencoder` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the sparsity penalty. |
| `EncodedSize` | Gets the size of the encoded representation (latent space). |
| `UseAuxiliaryLoss` | Gets or sets whether to use auxiliary loss (sparsity penalty) during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateLoss(Tensor<>,Tensor<>)` | Calculates the loss between the predicted output and the expected output. |
| `CalculateOutputGradient(Tensor<>,Tensor<>)` | Calculates the gradient of the loss function with respect to the network output. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for sparse autoencoders, which penalizes non-sparse activations. |
| `ComputeSparsityGradient` | Computes the gradient of the sparsity loss with respect to encoder activations. |
| `CreateNewInstance` | Creates a new instance of the autoencoder model. |
| `Decode(Tensor<>)` | Decodes the latent space representation back to the original space. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data for the Autoencoder. |
| `Encode(Tensor<>)` | Encodes the input data into the latent space representation. |
| `GenerateSamples(Int32,Double,Double)` | Generates new data samples by sampling points in the latent space and decoding them. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the sparsity loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Gets metadata about the autoencoder model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the autoencoder. |
| `PredictCore(Tensor<>)` | Makes a prediction using the autoencoder by encoding and then decoding the input. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data for the Autoencoder. |
| `SetSparsityParameter()` | Sets the target sparsity parameter for sparse autoencoder training. |
| `ShapesCompatibleIgnoringUnresolved(Int32[],Int32[])` | Validates that the custom layers conform to autoencoder requirements. |
| `Train(Tensor<>,Tensor<>)` | Trains the autoencoder on the provided data. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the autoencoder. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_averageActivation` | Stores the average activation level for diagnostics. |
| `_batchSize` | The size of each batch used in training. |
| `_epochs` | The number of training epochs. |
| `_lastEncoderActivations` | Stores the last encoder activations for auxiliary loss computation. |
| `_lastSparsityLoss` | Stores the last computed sparsity loss for diagnostics. |
| `_learningRate` | The learning rate used for training the autoencoder. |
| `_lossFunction` | The loss function used to measure the difference between the input and the reconstructed output. |
| `_sparsityParameter` | Target sparsity parameter (desired average activation level). |

