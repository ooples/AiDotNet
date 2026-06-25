---
title: "Cutie<T>"
description: "Cutie: Cutting-edge Video Instance Segmentation with transformer memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Segmentation`

Cutie: Cutting-edge Video Instance Segmentation with transformer memory.

## For Beginners

Cutie tracks objects in video. You provide a mask showing
where the object is in the first frame, and Cutie finds that same object in all
following frames - even when it moves, changes shape, or gets partially hidden.

Key features:

- Object permanence understanding (tracks objects even when briefly occluded)
- Efficient memory management for long videos
- High-quality mask predictions
- Multi-object tracking support

Example usage (native mode for training):

Example usage (ONNX mode for inference):

## How It Works

Cutie is designed for semi-supervised video object segmentation (VOS).
Given a mask for an object in the first frame, Cutie tracks and segments that object
throughout the entire video with high accuracy.

**Reference:** "Putting the Object Back into Video Object Segmentation"
https://arxiv.org/abs/2310.12982

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Cutie` | Creates a Cutie model using native layers for training and inference. |
| `Cutie(NeuralNetworkArchitecture<>,String,Int32,CutieOptions)` | Creates a Cutie model using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentMemoryCount` | Gets the current number of items in the memory bank. |
| `InputHeight` | Gets the input height. |
| `InputWidth` | Gets the input width. |
| `MemorySize` | Gets the maximum memory size. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddObject(Tensor<>,Tensor<>)` | Adds a new object to track by storing its features in memory. |
| `ClearMemory` | Clears the memory bank. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `PredictOnnx(Tensor<>)` | Performs inference using the ONNX model. |
| `SegmentFrame(Tensor<>)` | Segments a single frame using the current memory state. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `TrackObject(List<Tensor<>>,Tensor<>)` | Tracks and segments an object across video frames. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_inputChannels` | Number of input channels. |
| `_inputHeight` | Input frame height. |
| `_inputWidth` | Input frame width. |
| `_lossFunction` | The loss function for training. |
| `_memoryBank` | Memory bank storing key-value pairs for object tracking. |
| `_memorySize` | Maximum size of the memory bank. |
| `_numFeatures` | Feature dimension for the model. |
| `_onnxModelPath` | Path to the ONNX model file. |
| `_onnxSession` | The ONNX inference session for the model. |
| `_optimizer` | The optimizer used for training. |
| `_useNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |

