---
title: "TimeSformer<T>"
description: "TimeSformer: Is Space-Time Attention All You Need for Video Understanding?"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.ActionRecognition`

TimeSformer: Is Space-Time Attention All You Need for Video Understanding?

## For Beginners

TimeSformer is a transformer-based model for video classification
that applies attention across both space and time dimensions. Unlike CNNs that use
3D convolutions, TimeSformer uses pure self-attention to understand video content.

Key capabilities:

- Video action recognition (classify what action is happening)
- Temporal reasoning (understand events across time)
- Scene understanding (understand spatial context)

The model uses "divided space-time attention" where:

1. First, attention is applied across time (same spatial location, different frames)
2. Then, attention is applied across space (same frame, different locations)

Example usage (native mode for training):

Example usage (ONNX mode for inference only):

## How It Works

**Technical Details:**

- Divided space-time attention for efficiency
- Patch embedding similar to ViT
- Learnable positional embeddings for space and time
- Classification token for final prediction

**Reference:** "Is Space-Time Attention All You Need for Video Understanding?"
https://arxiv.org/abs/2102.05095

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSformer` | Creates a TimeSformer model using native layers for training and inference. |
| `TimeSformer(NeuralNetworkArchitecture<>,String,Int32,Int32,TimeSformerOptions)` | Creates a TimeSformer model using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionMode` | Gets the attention type used. |
| `EmbedDim` | Gets the embedding dimension. |
| `NumClasses` | Gets the number of output classes. |
| `NumFrames` | Gets the number of frames processed. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Tensor<>)` | Classifies video frames into action categories. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `ExtractFeatures(Tensor<>)` | Extracts video features before the classification head. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` |  |
| `GetTopKPredictions(Tensor<>,Int32)` | Gets the top-K predicted action classes with probabilities. |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

