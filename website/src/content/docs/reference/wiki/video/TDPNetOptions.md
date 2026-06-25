---
title: "TDPNetOptions"
description: "Configuration options for TDPNet temporal difference prediction network."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for TDPNet temporal difference prediction network.

## For Beginners

Instead of building the whole in-between frame from scratch, TDPNet
starts with a simple average of the two frames and then predicts what needs to change.
This is much easier (like correcting a rough draft vs writing from blank) and allows
a smaller, faster network to achieve good results.

## How It Works

TDPNet (2024) predicts temporal differences for efficient interpolation:

- Temporal difference prediction: instead of predicting the full intermediate frame, TDPNet

predicts only the temporal difference (residual) between the intermediate frame and a
linear blend of the two inputs, focusing network capacity on the non-trivial parts

- Difference-aware attention: self-attention modules that attend specifically to regions

where the temporal difference is large (motion boundaries, occlusions), avoiding wasting
computation on static regions where the linear blend is already accurate

- Coarse-to-fine difference refinement: multi-scale architecture where coarse-level

differences capture global motion corrections and fine-level differences add sharp
texture details and boundary refinement

- Lightweight backbone: because predicting residuals is easier than predicting full frames,

TDPNet can use a significantly lighter backbone while matching the quality of heavier
full-frame prediction methods

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TDPNetOptions` | Initializes a new instance with default values. |
| `TDPNetOptions(TDPNetOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DifferenceThreshold` | Gets or sets the difference threshold for sparse attention. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumDiffBlocks` | Gets or sets the number of difference prediction blocks. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads in difference-aware attention. |
| `NumScales` | Gets or sets the number of refinement scales. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

