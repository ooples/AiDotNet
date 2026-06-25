---
title: "SlowFast<T>"
description: "SlowFast Networks for Video Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.ActionRecognition`

SlowFast Networks for Video Recognition.

## For Beginners

SlowFast is a two-pathway network that processes video at two
different frame rates simultaneously:

- Slow pathway: Processes fewer frames (e.g., 4 fps) but with more channels to capture spatial details
- Fast pathway: Processes more frames (e.g., 32 fps) but with fewer channels to capture motion

This design is inspired by how human vision has:

- Parvo cells: Slow but detailed spatial processing
- Magno cells: Fast but coarse motion processing

Example usage:

## How It Works

**Technical Details:**

- Two-pathway design with lateral connections
- Slow pathway: T frames, C channels
- Fast pathway: alphaT frames, betaC channels (alpha=8, beta=1/8 typically)
- Lateral connections fuse information between pathways

**Reference:** "SlowFast Networks for Video Recognition" ICCV 2019
https://arxiv.org/abs/1812.03982

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SlowFast` | Initializes a new instance with default architecture settings. |
| `SlowFast(NeuralNetworkArchitecture<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,IActivationFunction<>,IReadOnlyList<ILayer<>>,IReadOnlyList<ILayer<>>,Int32,Int32,Int32,Int32,SlowFastOptions)` | Creates a SlowFast model using native layers for training and inference. |
| `SlowFast(NeuralNetworkArchitecture<>,String,Int32,IActivationFunction<>,Int32,Int32,Int32,Int32,SlowFastOptions)` | Creates a SlowFast model using a pretrained ONNX model for inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Tensor<>)` | Classifies video frames into action categories. |
| `ConcatenateTensors(Tensor<>,Tensor<>)` | Concatenates two tensors along the channel dimension. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes SlowFast-specific configuration data and reinitializes layers. |
| `Dispose(Boolean)` | Releases the unmanaged resources and optionally releases the managed resources. |
| `ForwardForTraining(Tensor<>)` | Tape-aware forward pass for training. |
| `GetModelMetadata` | Gets metadata about this model for serialization. |
| `GetOptions` |  |
| `GetTopKPredictions(Tensor<>,Int32)` | Gets top-K predictions with probabilities. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes SlowFast-specific configuration data including training component types. |
| `SplitGradient(Tensor<>)` | Splits the gradient tensor for slow and fast pathways. |
| `SubsampleFrames(Tensor<>,Int32)` | Subsamples frames by taking every n-th frame. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_customFastLayers` | Custom fast pathway layers provided by user (null = use default). |
| `_customFusionLayers` | Custom fusion layers provided by user (null = use default). |
| `_fastLayers` | Fast pathway layers (high temporal resolution, low channel capacity). |
| `_fusionLayers` | Fusion layers that combine slow and fast pathway outputs for classification. |
| `_slowLayerCount` | Index range markers into the unified Layers list. |

