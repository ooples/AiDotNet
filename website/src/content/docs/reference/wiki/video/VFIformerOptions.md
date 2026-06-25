---
title: "VFIformerOptions"
description: "Configuration options for VFIformer video frame interpolation transformer."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for VFIformer video frame interpolation transformer.

## For Beginners

VFIformer uses transformers (the same technology behind GPT and
other AI models) for frame interpolation. The attention mechanism lets every part of the
output frame "look at" relevant parts of both input frames, producing better results
especially for complex scenes.

## How It Works

VFIformer (Lu et al., CVPR 2022) applies vision transformers to frame interpolation:

- Cross-scale attention: transformer attention mechanism that attends across multiple feature

scales simultaneously, capturing both local fine-grained correspondences and global scene
structure in a single attention operation

- Flow-guided deformable attention: attention queries are positioned based on estimated

optical flow, so the model attends to motion-relevant regions rather than wasting attention
on irrelevant spatial locations

- Multi-frame transformer decoder: a transformer decoder that takes tokens from both input

frames and generates intermediate frame tokens, with causal masking adapted for spatial
rather than temporal ordering

- Efficient token design: uses feature pooling and stride patterns that reduce token count

by 16x compared to naive patch tokenization, enabling high-resolution processing

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VFIformerOptions` | Initializes a new instance with default values. |
| `VFIformerOptions(VFIformerOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDecoderLayers` | Gets or sets the number of transformer decoder layers. |
| `NumDeformablePoints` | Gets or sets the number of deformable attention points. |
| `NumEncoderLayers` | Gets or sets the number of transformer encoder layers. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

