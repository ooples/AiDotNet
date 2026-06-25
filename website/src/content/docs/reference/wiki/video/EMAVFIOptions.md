---
title: "EMAVFIOptions"
description: "Configuration options for the EMA-VFI swin-based inter-frame attention model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the EMA-VFI swin-based inter-frame attention model.

## For Beginners

Instead of explicitly computing optical flow (how pixels move),
EMA-VFI uses attention to simultaneously figure out "what moved where" (motion) and
"what does it look like" (appearance). By processing both together, it avoids errors
from bad flow estimates and produces cleaner interpolated frames.

## How It Works

EMA-VFI (Zhang et al., CVPR 2023) extracts motion and appearance via inter-frame attention:

- Swin-based cross-attention: shifted window cross-attention between frame pairs to extract

motion correspondence without explicit optical flow computation

- Dual-branch extraction: one branch captures motion dynamics (displacement features)

while the other captures appearance information (texture, color) from both frames

- Bilateral motion estimation: bidirectional motion fields estimated simultaneously

using cross-attention scores as soft correspondence weights

- Multi-scale feature fusion: hierarchical feature pyramid with cross-scale connections

for handling both small and large motions

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EMAVFIOptions` | Initializes a new instance with default values. |
| `EMAVFIOptions(EMAVFIOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BidirectionalMotion` | Gets or sets whether to use bidirectional motion estimation. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels in the encoder. |
| `NumHeads` | Gets or sets the number of attention heads in each swin block. |
| `NumScales` | Gets or sets the number of pyramid scales for multi-scale fusion. |
| `NumSwinBlocks` | Gets or sets the number of swin cross-attention blocks per scale. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |
| `WindowSize` | Gets or sets the swin window size for local attention. |

