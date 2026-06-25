---
title: "DualXVSR<T>"
description: "DualX-VSR: dual axial spatial-temporal transformer without motion compensation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

DualX-VSR: dual axial spatial-temporal transformer without motion compensation.

## For Beginners

Most video SR models need to figure out how objects moved between
frames (optical flow). DualX-VSR skips this step by using a clever attention pattern
that looks along two crossing axes simultaneously. Like reading a crossword puzzle by
checking both across and down -- you understand the full picture without tracing each
letter's path.

**Usage:**

## How It Works

DualX-VSR (2025) eliminates explicit motion compensation through dual axial attention:

- Dual axial attention: decomposes 3D attention into two orthogonal axis pairs

(height-temporal and width-temporal), capturing full spatial-temporal context with
linear complexity instead of cubic

- Motion-free alignment: the crossed axial attention patterns implicitly capture

inter-frame correspondence without computing optical flow or deformable offsets

- Symmetric bidirectional propagation: features propagate both forward and backward

in time with shared axial attention weights

- Each dual axial block performs height-temporal attention followed by width-temporal

attention, ensuring every position can attend to any other position in the 3D volume
through the composition of two axis-aligned operations

**Reference:** "DualX-VSR: Dual Axial Spatial-Temporal Transformer for
Real-World Video Super-Resolution without Motion Compensation" (2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DualXVSR(NeuralNetworkArchitecture<>,DualXVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DualX-VSR model in native training mode. |
| `DualXVSR(NeuralNetworkArchitecture<>,String,DualXVSROptions)` | Creates a DualX-VSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

