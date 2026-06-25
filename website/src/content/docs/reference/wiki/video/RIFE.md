---
title: "RIFE<T>"
description: "Real-time Intermediate Flow Estimation (RIFE) for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

Real-time Intermediate Flow Estimation (RIFE) for video frame interpolation.

## For Beginners

RIFE is a state-of-the-art model for generating intermediate frames between
two existing video frames. This is useful for:

- Increasing video frame rate (e.g., 24fps to 60fps)
- Creating slow-motion effects
- Smoothing video playback
- Reducing temporal aliasing

RIFE uses a privileged distillation approach with intermediate flow estimation
to create realistic frames at arbitrary positions between input frames.

## How It Works

**Technical Details:**

- Uses IFNet architecture for intermediate flow estimation
- Coarse-to-fine flow refinement across multiple scales
- Context-aware fusion with feature maps
- Supports arbitrary timestep interpolation (not just midpoint)

**Reference:** Huang et al., "RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation"
ECCV 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RIFE` | Initializes a new instance of the RIFE class. |

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
| `AccumulateInputGradients(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Accumulates gradients from multiple branches back to the input. |
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise. |
| `BilinearDownsample(Tensor<>,Int32)` | Downsamples a tensor using bilinear interpolation (backward of upsample). |
| `ClearActivationCache` | Clears the activation cache. |
| `CombineFlowGradients(Tensor<>,Tensor<>,Tensor<>)` | Combines flow gradients from different sources. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `ExtractLayerReferences` | (Re)builds the sub-list references (`_encoder`, `_flowDecoder`, etc.) that `Double)` uses, from the canonical `Layers` list. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetPixelSafe(Tensor<>,Int32,Int32,Int32,Int32,Int32,Int32)` | Gets a pixel value safely with boundary handling. |
| `InitializeLayers` |  |
| `Interpolate(Tensor<>,Tensor<>,Double)` | Interpolates frames between two input frames. |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SplitConcatenatedGradient(Tensor<>,Int32,Int32)` | Splits a concatenated gradient tensor along the channel dimension. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |
| `UpsampleFrameRate(List<Tensor<>>,Int32)` | Interpolates multiple frames between input frames for frame rate upsampling. |
| `WarpBackward(Tensor<>,Tensor<>,Tensor<>)` | Computes gradients through the warping operation. |

