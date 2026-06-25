---
title: "BasicVSRPlusPlus<T>"
description: "BasicVSR++ (Basic Video Super-Resolution++) for temporal video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

BasicVSR++ (Basic Video Super-Resolution++) for temporal video super-resolution.

## For Beginners

BasicVSR++ is a video super-resolution model that upscales
low-resolution videos to higher resolution while maintaining temporal consistency.
Unlike single-image methods (like RealESRGAN), it uses information from multiple
frames to produce sharper and more consistent results.

Key concepts:

1. **Bidirectional Propagation:** Uses both past and future frames to enhance

the current frame, ensuring temporal coherence.

2. **Optical Flow:** Estimates how pixels move between frames to align features.
3. **Deformable Alignment:** Uses learned offsets to precisely align features

even with complex motions.

Example usage:

## How It Works

BasicVSR++ improves upon BasicVSR with:

- Second-order grid propagation for better temporal modeling
- Flow-guided deformable alignment for accurate feature alignment
- Bidirectional propagation for utilizing both past and future frames

**Reference:** Chan et al., "BasicVSR++: Improving Video Super-Resolution with
Enhanced Propagation and Alignment", CVPR 2022. https://arxiv.org/abs/2104.13371

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BasicVSRPlusPlus` | Initializes a new instance with default architecture settings. |
| `BasicVSRPlusPlus(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Double,BasicVSRPlusPlusOptions)` | Creates a BasicVSR++ model for native training and inference. |
| `BasicVSRPlusPlus(NeuralNetworkArchitecture<>,String,Int32,Int32,Int32,Int32,BasicVSRPlusPlusOptions)` | Creates a BasicVSR++ model using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumFeatures` | Gets the number of feature channels. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UpscaleFactor` | Gets the upscaling factor for this model. |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateGradient(Tensor<>,Tensor<>)` | Accumulates gradient values into a target tensor. |
| `BidirectionalPropagationWithCache(List<Tensor<>>,List<ValueTuple<Tensor<>,Tensor<>>>,Int32)` | Performs bidirectional propagation with caching of all intermediate activations for proper gradient computation during backward pass. |
| `ClearActivationCache` | Clears all activation caches. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EnhanceVideo(Tensor<>)` | Enhances a sequence of video frames using temporal super-resolution. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SplitConcatenatedGradient(Tensor<>,Int32,Int32)` | Splits a gradient tensor at a concatenation point along the channel dimension. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `Upscale(Tensor<>)` |  |
| `WarpBackward(Tensor<>,Tensor<>,Tensor<>)` | Backward pass through bilinear warping operation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_backwardAlignments` | Deformable alignment modules for backward propagation. |
| `_backwardConvs` | Propagation convolutions for backward pass. |
| `_cachedFeatures` | Cached features from last forward pass for training. |
| `_featExtract` | Feature extraction layer at the input. |
| `_flowEstimator` | The SPyNet flow estimator for optical flow between frames. |
| `_forwardAlignments` | Deformable alignment modules for forward propagation. |
| `_forwardConvs` | Propagation convolutions for forward pass. |
| `_learningRate` | Learning rate for training. |
| `_numFeatures` | Number of feature channels in the network. |
| `_numPropagations` | Number of propagation iterations per direction. |
| `_numResidualBlocks` | Number of residual blocks. |
| `_onnxModelPath` | Path to the ONNX model file. |
| `_onnxSession` | The ONNX inference session for the model. |
| `_outputConv` | Final convolution for output reconstruction. |
| `_residualBlocks` | Residual blocks for feature extraction. |
| `_scaleFactor` | The upscaling factor (2 or 4). |
| `_upsampleLayers` | Upsampling layers using PixelShuffle. |
| `_useNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |

