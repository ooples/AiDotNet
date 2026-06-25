---
title: "SiameseNetwork<T>"
description: "Implements a Siamese Neural Network for comparing pairs of inputs and determining their similarity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements a Siamese Neural Network for comparing pairs of inputs and determining their similarity.

## For Beginners

A Siamese Network is a special type of neural network designed to compare two inputs
and determine how similar they are to each other.

Imagine you have two photos and want to know if they show the same person. A Siamese Network
processes both photos through identical neural networks (like twins, hence the name "Siamese"),
creates a compact representation (called an "embedding") of each photo, and then compares these
representations to determine similarity.

Common applications include:

- Face recognition (are these two faces the same person?)
- Signature verification (is this signature authentic?)
- Document similarity (how similar are these two texts?)
- Product recommendations (finding similar products)

The key advantage of Siamese Networks is that they can learn to recognize similarity even for
inputs they've never seen before during training.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SiameseNetwork` | Initializes a new instance of the SiameseNetwork class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the contrastive auxiliary loss. |
| `ContrastiveMargin` | Gets or sets the margin for contrastive loss. |
| `ParameterCount` | Gets the total number of trainable parameters in the Siamese network. |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (contrastive/triplet loss) should be used during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineEmbeddings(Vector<>,Vector<>)` | Combines two embedding vectors into a single vector for comparison. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss (contrastive loss) for similarity learning. |
| `CreateNewInstance` | Creates a new instance of the Siamese network with the same architecture. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Siamese network-specific data from a binary reader. |
| `ForwardForTraining(Tensor<>)` | Defines the Siamese forward graph for tape-based training. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the contrastive auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Gets metadata about the Siamese Network. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the neural network. |
| `PredictCore(Tensor<>)` | Makes a prediction using the Siamese network to compare the similarity between inputs. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Siamese network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the Siamese network on pairs of inputs with their expected similarity. |
| `UpdateParameters(Vector<>)` | Updates the network parameters with new values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedEmbeddingPairs` | Cache for embedding pairs and their similarity labels during training. |
| `_outputLayer` | The final layer that compares the embeddings and produces a similarity score. |
| `_subnetwork` | The shared neural network that processes each input independently. |

