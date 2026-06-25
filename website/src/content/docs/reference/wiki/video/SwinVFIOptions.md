---
title: "SwinVFIOptions"
description: "Configuration options for SwinVFI Swin Transformer-based video frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for SwinVFI Swin Transformer-based video frame interpolation.

## For Beginners

SwinVFI uses the Swin Transformer (a powerful attention-based
architecture) to look at both input frames simultaneously and figure out what goes between
them, without needing to estimate motion explicitly. The "shifted window" approach makes it
efficient enough to handle full-resolution video frames.

## How It Works

SwinVFI (2022) applies Swin Transformer architecture to frame interpolation:

- Swin Transformer encoder: uses shifted-window self-attention to encode input frame pairs

with linear complexity (O(N) vs O(N^2) for full attention), enabling processing of
high-resolution frames without excessive memory

- Cross-frame window attention: extends Swin's shifted-window mechanism to cross-attend

between features from the two input frames, capturing inter-frame correspondences within
each local window and globally through window shifting

- Hierarchical feature pyramid: multi-scale feature extraction with Swin blocks at each

level, capturing both fine-grained texture details and large-scale motion context

- Flow-free synthesis: directly synthesizes the intermediate frame from cross-attended

features without explicit optical flow estimation, avoiding flow-related artifacts

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SwinVFIOptions` | Initializes a new instance with default values. |
| `SwinVFIOptions(SwinVFIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumStages` | Gets or sets the number of hierarchical stages. |
| `NumSwinBlocks` | Gets or sets the number of Swin Transformer blocks per stage. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |
| `WindowSize` | Gets or sets the window size for shifted-window attention. |

