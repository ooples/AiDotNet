---
title: "UPRNet<T>"
description: "UPR-Net: Unified Pyramid Recurrent Network for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

UPR-Net: Unified Pyramid Recurrent Network for video frame interpolation.

## For Beginners

UPR-Net builds a "pyramid" of progressively-smaller
versions of each input frame, then estimates motion (optical flow) at the
smallest scale and progressively refines it as it works back up to full
resolution. The same recurrent module is applied at each scale, which
keeps the model lightweight while still being accurate.

## How It Works

**References:**

- Paper: "A Unified Pyramid Recurrent Network for Video Frame Interpolation" (Jin et al., CVPR 2023, arXiv:2211.03456)

Architecture (per Jin et al. 2023 §3):

Implementation notes for this port: the bilinear warp is implemented inline
via direct grid-sample arithmetic on the tensor (no warp layer in the
library), and the per-level ConvLSTM hidden state is reset between forward
passes since training treats each (frame0, frame1) pair as an independent
sequence. Full forward/backward consistency checking is included via the
recurrent-refinement loop (Jin et al. §3.4).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UPRNet(NeuralNetworkArchitecture<>,String,UPRNetOptions)` | Creates a UPR-Net model for ONNX inference. |
| `UPRNet(NeuralNetworkArchitecture<>,UPRNetOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a UPR-Net model for native training and inference. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BilinearUpsample(Tensor<>,Int32,Int32,Boolean)` | Bilinear upsample of an NCHW tensor to a target spatial resolution. |
| `BilinearWarp(Tensor<>,Tensor<>,Boolean)` | Bilinear backward-warp of an NCHW feature tensor by an NCHW flow field. |
| `ConcatChannels(Tensor<>[])` | Concatenates a sequence of NCHW tensors along the channel axis. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `Forward(Tensor<>)` | UPR-Net forward pass: encode pyramid → coarse-to-fine refinement → output. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PreprocessFrames(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SliceChannels(Tensor<>,Int32,Int32)` | Slices a contiguous range of channels from an NCHW tensor. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

