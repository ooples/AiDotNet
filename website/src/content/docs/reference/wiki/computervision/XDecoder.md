---
title: "XDecoder<T>"
description: "X-Decoder: Generalized Decoding for Pixel, Image, and Language."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Segmentation.Foundation`

X-Decoder: Generalized Decoding for Pixel, Image, and Language.

## For Beginners

X-Decoder is a generalist model that simultaneously handles referring
segmentation (find and segment an object from a text description), open-vocabulary segmentation
(segment objects from any text class list), and image captioning — all with one shared decoder.
It bridges the gap between pixel-level understanding and language understanding.

Common use cases:

- Referring segmentation ("segment the red car on the left")
- Open-vocabulary semantic segmentation with arbitrary class names
- Image captioning and visual question answering
- Multi-modal vision-language systems

## How It Works

**Technical Details:**

- Two-path decoder: pixel path (mask predictions) and token path (text predictions)
- Both paths share the same cross-attention mechanism
- Supports any combination of text, pixel, and image inputs/outputs
- Backbone: Focal-T/B/L transformer
- Single model handles 7+ vision-language tasks

**Reference:** Zou et al., "Generalized Decoding for Pixel, Image, and Language", CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XDecoder(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,XDecoderModelSize,Double,XDecoderOptions)` | Initializes X-Decoder in native (trainable) mode. |
| `XDecoder(NeuralNetworkArchitecture<>,String,Int32,Int32,XDecoderModelSize,XDecoderOptions)` | Initializes X-Decoder in ONNX (inference-only) mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether this X-Decoder instance supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new X-Decoder instance with the same configuration but fresh weights. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Reads X-Decoder configuration from a binary stream. |
| `Dispose(Boolean)` | Releases managed resources including the ONNX inference session. |
| `GetModelMetadata` | Collects metadata describing this X-Decoder model's configuration. |
| `GetOptions` | Gets the configuration options for this X-Decoder model. |
| `InitializeLayers` | Initializes the encoder and dual-path decoder layers for X-Decoder. |
| `PredictCore(Tensor<>)` | Runs a forward pass through X-Decoder for generalist vision-language segmentation. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Writes X-Decoder configuration to a binary stream. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step. |
| `UpdateParameters(Vector<>)` | Updates all trainable parameters from a flat parameter vector. |

