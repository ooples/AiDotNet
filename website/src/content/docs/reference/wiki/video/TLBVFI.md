---
title: "TLBVFI<T>"
description: "TLBVFI: token-level bidirectional video frame interpolation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.FrameInterpolation`

TLBVFI: token-level bidirectional video frame interpolation.

## For Beginners

TLBVFI works with small patches (tokens) instead of individual pixels,
like reading words instead of individual letters. It matches patches in both frames and
uses them to build the intermediate frame efficiently.

**Usage:**

## How It Works

TLBVFI (2024) operates at the token level for bidirectional frame interpolation:

- Token-level processing: divides frames into non-overlapping tokens and performs all

operations at the token level for efficient transformer-based processing

- Bidirectional token matching: finds corresponding tokens in both input frames using

learned cross-attention, handling forward and backward motion in a single pass

- Token-level flow: estimates optical flow at the patch level for robustness and efficiency,

with sub-token refinement after initial matching

- Adaptive token merging: dynamically merges tokens in low-motion regions while keeping

fine-grained tokens in high-motion areas

**Reference:** "TLBVFI: Token-Level Bidirectional Video Frame Interpolation" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TLBVFI(NeuralNetworkArchitecture<>,String,TLBVFIOptions)` | Creates a TLBVFI model in ONNX inference mode. |
| `TLBVFI(NeuralNetworkArchitecture<>,TLBVFIOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a TLBVFI model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Interpolate(Tensor<>,Tensor<>,Double)` |  |

