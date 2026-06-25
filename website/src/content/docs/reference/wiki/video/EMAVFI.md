---
title: "EMAVFI<T>"
description: "EMA-VFI: extracting motion and appearance via inter-frame attention for video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

EMA-VFI: extracting motion and appearance via inter-frame attention for video frame interpolation.

## For Beginners

Instead of first computing optical flow (how pixels move) and then
warping frames, EMA-VFI uses attention to simultaneously figure out "what moved where"
(motion) and "what does it look like" (appearance). By processing both together, it avoids
errors from bad flow estimates and produces cleaner, sharper interpolated frames.

**Usage:**

## How It Works

EMA-VFI (Zhang et al., CVPR 2023) uses swin-based cross-attention for motion and appearance:

- Swin cross-attention: shifted window cross-attention between frame pairs extracts dense

motion correspondence without explicit optical flow computation, avoiding flow estimation
errors that plague traditional methods

- Dual-branch extraction: motion branch captures displacement features (where things moved)

while appearance branch captures texture and color information (what things look like),
fused via learned gating for each pixel

- Bilateral motion estimation: bidirectional motion fields estimated simultaneously using

cross-attention scores as soft correspondence weights, naturally handling occlusion

- Multi-scale feature fusion: hierarchical feature pyramid with cross-scale connections

handles both small sub-pixel motions and large displacements across frames

**Reference:** "Extracting Motion and Appearance via Inter-Frame Attention for Efficient
Video Frame Interpolation" (Zhang et al., CVPR 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EMAVFI(NeuralNetworkArchitecture<>,EMAVFIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an EMA-VFI model in native training mode. |
| `EMAVFI(NeuralNetworkArchitecture<>,String,EMAVFIOptions)` | Creates an EMA-VFI model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

