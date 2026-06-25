---
title: "StreamDiffVSR<T>"
description: "Stream-DiffVSR: causally-conditioned diffusion for online video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Enhancement`

Stream-DiffVSR: causally-conditioned diffusion for online video super-resolution.

## For Beginners

Most video upscalers need to see future frames, which adds delay.
Stream-DiffVSR only uses past frames, making it suitable for live streaming or
video calls where latency matters.

**Usage:**

## How It Works

Stream-DiffVSR (Li et al., 2025) achieves low-latency online video SR through:

- Auto-regressive temporal guidance using previously generated HR frames
- A 4-step distilled denoiser (compressed from ~50 diffusion steps)
- Causal temporal conditioning (past frames only, no future lookahead)

This enables streaming 4x video super-resolution with temporal consistency.

**Reference:** "Stream-DiffVSR: Low-Latency Streamable Diffusion-based Video Super-Resolution"
(Li et al., 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StreamDiffVSR(NeuralNetworkArchitecture<>,StreamDiffVSROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a StreamDiffVSR model in native training mode. |
| `StreamDiffVSR(NeuralNetworkArchitecture<>,String,StreamDiffVSROptions)` | Creates a StreamDiffVSR model in ONNX inference mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Upscale(Tensor<>)` |  |

