---
title: "XVFI<T>"
description: "XVFI extreme video frame interpolation for 4K/8K content with very large motion."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

XVFI extreme video frame interpolation for 4K/8K content with very large motion.

## For Beginners

XVFI is designed for extreme cases: very high resolution video (4K/8K)
where objects move very far between frames. It uses a multi-level approach that first captures
big movements, then progressively adds fine detail, enabling frame interpolation even when
objects move hundreds of pixels between frames.

## How It Works

**References:**

- Paper: "XVFI: eXtreme Video Frame Interpolation" (Sim et al., ICCV 2021)

XVFI handles extreme motion for high-FPS video with several key innovations:

- Extreme motion handling: designed for 4K/8K video with very large frame-to-frame

displacements (100+ pixels), far beyond what standard flow networks can handle

- Complementary flow: estimates both global (affine) and local (dense) optical flow fields,

combining them with learned blending weights so global flow handles camera motion and
local flow handles object motion

- Multi-scale architecture: a 7-level feature pyramid with flow estimation at each scale,

starting from 1/64 resolution for very large motions and refining up to full resolution

- Bilinear flow upsampling: uses learned bilinear upsampling kernels (not fixed bilinear

interpolation) to upsample flow fields between pyramid levels, preserving sharp motion
boundaries during upsampling

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XVFI(NeuralNetworkArchitecture<>,String,XVFIOptions)` | Creates an XVFI model for ONNX inference. |
| `XVFI(NeuralNetworkArchitecture<>,XVFIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an XVFI model for native training and inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `Interpolate(Tensor<>,Tensor<>,Double)` | Interpolates between two frames at timestep t. |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

