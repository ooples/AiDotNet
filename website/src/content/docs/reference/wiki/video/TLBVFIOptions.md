---
title: "TLBVFIOptions"
description: "Configuration options for TLBVFI token-level bidirectional video frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for TLBVFI token-level bidirectional video frame interpolation.

## For Beginners

TLBVFI breaks each frame into small patches (tokens) and works with
these patches instead of individual pixels. This is faster and more robust, like reading
words instead of individual letters. It finds matching patches in both frames and uses them
to build the intermediate frame.

## How It Works

TLBVFI (2024) operates at the token level for bidirectional frame interpolation:

- Token-level processing: divides input frames into non-overlapping tokens (patches) and

performs all operations (flow estimation, feature extraction, synthesis) at the token level,
enabling efficient processing of high-resolution frames with transformer architectures

- Bidirectional token matching: for each target-time token, finds corresponding tokens in

both input frames using learned token-level cross-attention, naturally handling both forward
and backward motion in a single pass

- Token-level flow: estimates optical flow at the token (patch) level rather than the pixel

level, which is more robust to noise and local ambiguities while being computationally
cheaper, with sub-token refinement applied after initial matching

- Adaptive token merging: dynamically merges tokens in low-motion regions to reduce

computation, while keeping fine-grained tokens in high-motion areas for accuracy

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TLBVFIOptions` | Initializes a new instance with default values. |
| `TLBVFIOptions(TLBVFIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads for token matching. |
| `NumMatchingBlocks` | Gets or sets the number of bidirectional matching blocks. |
| `NumSynthesisBlocks` | Gets or sets the number of synthesis refinement blocks. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `TokenSize` | Gets or sets the token (patch) size in pixels. |
| `Variant` | Gets or sets the model variant. |

