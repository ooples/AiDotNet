---
title: "VRT<T>"
description: "VRT: A Video Restoration Transformer for video super-resolution, deblurring, and denoising."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Restoration`

VRT: A Video Restoration Transformer for video super-resolution, deblurring, and denoising.

## For Beginners

VRT improves video quality by analyzing multiple frames
together. Unlike image restoration that processes one frame at a time, VRT
uses temporal information to produce better, more consistent results.

Example usage (native mode for training):

Example usage (ONNX mode for inference only):

## How It Works

VRT (Video Restoration Transformer) is a powerful architecture for video restoration tasks:

- Video super-resolution (increasing video resolution)
- Video deblurring (removing motion blur)
- Video denoising (removing noise from videos)

**Reference:** "VRT: A Video Restoration Transformer"
https://arxiv.org/abs/2201.12288

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VRT` | Initializes a new instance with default architecture settings. |
| `VRT(NeuralNetworkArchitecture<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Int32,Int32,Int32,Int32,VRTOptions)` | Creates a VRT model using native layers for training and inference. |
| `VRT(NeuralNetworkArchitecture<>,String,Int32,VRTOptions)` | Creates a VRT model using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `Deblur(Tensor<>)` | Performs video deblurring. |
| `Denoise(Tensor<>)` | Performs video denoising. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PredictOnnx(Tensor<>)` | Performs inference using the ONNX model. |
| `PreprocessFrames(Tensor<>)` |  |
| `Restore(Tensor<>)` | Restores a video frame using the VRT model. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SuperResolve(Tensor<>)` | Performs video super-resolution. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `Upscale(Tensor<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_embedDim` | Embedding dimension for the model. |
| `_inputHeight` | Input frame height. |
| `_inputWidth` | Input frame width. |
| `_lossFunction` | The loss function for training. |
| `_numBlocks` | Number of transformer blocks. |
| `_numFrames` | Number of temporal frames processed together. |
| `_onnxModelPath` | Path to the ONNX model file. |
| `_onnxSession` | The ONNX inference session for the model. |
| `_optimizer` | The optimizer used for training. |
| `_scaleFactor` | Upscaling factor for super-resolution. |
| `_useNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |

