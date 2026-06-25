---
title: "ProPainter<T>"
description: "ProPainter for video inpainting - removes unwanted objects and fills regions in video."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Inpainting`

ProPainter for video inpainting - removes unwanted objects and fills regions in video.

## For Beginners

ProPainter is a state-of-the-art model for video inpainting,
which means removing unwanted objects from video and filling the resulting holes
with realistic content. This is useful for:

- Removing watermarks or logos from video
- Removing unwanted people or objects
- Repairing damaged video footage
- Creating special effects

Unlike image inpainting, video inpainting needs to maintain temporal consistency
across frames to avoid flickering and artifacts.

## How It Works

**Technical Details:**

- Dual-domain propagation (image + flow domains)
- Recurrent flow completion for temporal consistency
- Mask-guided sparse convolution
- Transformer-based global feature aggregation

**Reference:** Zhou et al., "ProPainter: Improving Propagation and Transformer for Video Inpainting"
ICCV 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProPainter` | Initializes a new instance of the ProPainter class. |
| `ProPainter(NeuralNetworkArchitecture<>,Int32,Int32,Int32,ProPainterOptions)` |  |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input channels. |
| `InputHeight` | Gets the input height for frames. |
| `InputWidth` | Gets the input width for frames. |
| `SupportsTraining` | Gets whether training is supported. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Element-wise addition of two tensors (residual connection). |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `Inpaint(List<Tensor<>>,List<Tensor<>>)` | Inpaints (fills) masked regions in video frames. |
| `Inpaint(Tensor<>,Tensor<>)` |  |
| `LayerNorm(Tensor<>)` | Applies layer normalization (standardization) to input tensor. |
| `MultiHeadSelfAttention(Tensor<>,Int32[])` | Applies multi-head self-attention following the Transformer architecture. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `RemoveObject(List<Tensor<>>,Tensor<>)` | Removes an object specified by mask from video frames. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

