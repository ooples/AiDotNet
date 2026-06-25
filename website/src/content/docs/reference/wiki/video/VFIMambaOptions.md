---
title: "VFIMambaOptions"
description: "Configuration options for VFIMamba state-space model for video frame interpolation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for VFIMamba state-space model for video frame interpolation.

## For Beginners

VFIMamba uses a new type of AI architecture called "Mamba" that can
process long sequences efficiently (unlike transformers which get slow with long inputs).
This means it can handle high-resolution frames without running out of memory, while still
understanding how every pixel relates to every other pixel.

## How It Works

VFIMamba (2024) applies Mamba (selective state space model) to frame interpolation:

- Selective state space: uses Mamba's selective scan mechanism instead of attention, achieving

linear complexity O(N) for processing frame features while maintaining global context,
enabling efficient processing of high-resolution frames

- Bidirectional scanning: scans frame features in both forward (left-to-right, top-to-bottom)

and backward directions, ensuring each pixel has full global context from all directions

- Cross-frame state propagation: the SSM state from one frame is propagated to condition

the processing of the other frame, enabling implicit motion correspondence without
explicit flow estimation

- Multi-scale Mamba blocks: hierarchical Mamba blocks at different spatial scales, capturing

both fine-grained texture details and coarse motion patterns

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VFIMambaOptions` | Initializes a new instance with default values. |
| `VFIMambaOptions(VFIMambaOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `ExpansionFactor` | Gets or sets the SSM expansion factor. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumMambaBlocks` | Gets or sets the number of Mamba blocks per stage. |
| `NumStages` | Gets or sets the number of hierarchical stages. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `StateDim` | Gets or sets the SSM state dimension. |
| `Variant` | Gets or sets the model variant. |

