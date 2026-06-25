---
title: "AnimateDiff<T>"
description: "AnimateDiff: Motion module for animating text-to-image diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Generation`

AnimateDiff: Motion module for animating text-to-image diffusion models.

## For Beginners

AnimateDiff makes still image generators create videos.
It plugs into existing models like Stable Diffusion to add movement.
Instead of generating one image, it generates multiple frames that flow smoothly.

Example usage (native mode for training):

Example usage (ONNX mode for inference only):

## How It Works

AnimateDiff is a motion module that:

- Adds temporal coherence to image diffusion models
- Converts image generators into video generators
- Learns motion patterns from video data

**Reference:** "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models"
https://arxiv.org/abs/2307.04725

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnimateDiff` | Creates an AnimateDiff model using native layers for training and inference. |
| `AnimateDiff(NeuralNetworkArchitecture<>,String,Int32,AnimateDiffOptions)` | Creates an AnimateDiff model using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputChannels` | Gets the number of input channels. |
| `NumFrames` | Gets the number of frames processed. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddMotion(Tensor<>)` | Adds motion to static features from an image diffusion model. |
| `BlendFeatures(Tensor<>,Tensor<>,Double)` | Blends motion module output with original features. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PredictCore(Tensor<>)` |  |
| `PredictOnnx(Tensor<>)` | Performs inference using the ONNX model. |
| `ProcessMotion(Tensor<>)` | Processes temporal features for motion modeling. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_featureHeight` | Feature height. |
| `_featureWidth` | Feature width. |
| `_inputChannels` | Number of input feature channels. |
| `_lossFunction` | The loss function for training. |
| `_numFrames` | Number of video frames. |
| `_numLayers` | Number of motion transformer layers. |
| `_onnxModelPath` | Path to the ONNX model file. |
| `_onnxSession` | The ONNX inference session for the model. |
| `_optimizer` | The optimizer used for training. |
| `_useNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |

