---
title: "VisionTransformer<T>"
description: "Implements the Vision Transformer (ViT) architecture for image classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Implements the Vision Transformer (ViT) architecture for image classification tasks.

## For Beginners

The Vision Transformer (ViT) is a modern approach to understanding images using transformers.

Unlike traditional neural networks that process images pixel by pixel or with sliding windows (convolutions),
ViT treats an image like a sentence of words:

- First, it cuts the image into small square patches (like breaking a sentence into words)
- Each patch gets converted to a numerical representation (like word embeddings)
- Position information is added so the model knows where each patch came from
- A special classification token is added to gather information about the whole image
- Transformer layers process all patches together, learning relationships between them
- Finally, the classification token's output is used to predict the image class

This approach has been very successful and often outperforms traditional convolutional neural networks,
especially when trained on large datasets.

## How It Works

The Vision Transformer applies transformer architecture, originally designed for natural language processing,
to computer vision tasks. It divides images into fixed-size patches, linearly embeds them, adds positional
embeddings, and processes the sequence through transformer encoder layers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VisionTransformer` | Creates a new Vision Transformer with the specified configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of parameters in the model. |
| `SupportsTraining` | Indicates whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Clone via fresh-construct + UpdateParameters rather than the default serialize/deserialize roundtrip. |
| `CreateNewInstance` | Creates a new instance of the Vision Transformer. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Vision Transformer-specific data. |
| `ForwardForTraining(Tensor<>)` | Tape-tracked forward pass for training. |
| `GetExtraTrainableTensors` | Surfaces `_clsToken` and `_positionalEmbeddings` to the tape training path. |
| `GetModelMetadata` | Gets the model metadata. |
| `GetOptions` |  |
| `GetParameters` | Gets all model parameters in a single vector. |
| `InitializeClassificationToken` | Initializes the classification token with random values. |
| `InitializeLayers` | Initializes the layers of the Vision Transformer. |
| `InitializePositionalEmbeddings` | Initializes the positional embeddings with random values. |
| `PredictCore(Tensor<>)` | Makes a prediction using the Vision Transformer. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Vision Transformer-specific data. |
| `Train(Tensor<>,Tensor<>)` | Trains the Vision Transformer on a single input-output pair. |
| `UpdateParameters(Vector<>)` | Updates the network's parameters with new values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_channels` | The number of color channels in input images. |
| `_classificationHead` | The final classification head (MLP). |
| `_clsToken` | The classification token embedding (rank-1 [hiddenDim] tensor). |
| `_hiddenDim` | The dimension of the embedding vectors. |
| `_imageHeight` | The height of input images. |
| `_imageWidth` | The width of input images. |
| `_mlpDim` | The dimension of the feed-forward network in each transformer layer. |
| `_numClasses` | The number of output classes. |
| `_numHeads` | The number of attention heads in each transformer layer. |
| `_numLayers` | The number of transformer encoder layers. |
| `_numPatches` | The total number of patches. |
| `_patchSize` | The size of each square patch. |
| `_positionalEmbeddings` | The positional embeddings (rank-2 [_numPatches+1, hiddenDim] tensor). |

