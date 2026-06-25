---
title: "MoMoOptions"
description: "Configuration options for MoMo momentum diffusion model for bi-directional flow."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for MoMo momentum diffusion model for bi-directional flow.

## For Beginners

Most frame interpolation methods estimate motion (optical flow) using
direct prediction, which can be blurry at object edges. MoMo instead uses a generative AI
model to create sharper, more accurate motion fields, then uses those flows to produce
the intermediate frame. Think of it as using AI to "draw" better motion maps.

## How It Works

MoMo (2024) is the first diffusion model for bi-directional optical flow in VFI:

- Flow diffusion: instead of directly regressing optical flow from a CNN (which produces

over-smoothed flow at boundaries), MoMo uses a denoising diffusion model to generate
bi-directional flow fields, capturing sharper motion boundaries and multi-modal flow

- Momentum-based flow modeling: incorporates a momentum prior that biases flow generation

toward physically plausible motions, reducing artifacts from unrealistic flow predictions

- Joint bi-directional generation: generates forward (t0 to t) and backward (t1 to t)

flows simultaneously in a single diffusion process, ensuring temporal consistency between
the two flow fields

- Flow-to-frame synthesis: the generated flows are used for backward warping with learned

occlusion masks and residual refinement to produce the final interpolated frame

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MoMoOptions` | Initializes a new instance with default values. |
| `MoMoOptions(MoMoOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `MomentumCoefficient` | Gets or sets the momentum coefficient for the flow prior. |
| `NumDiffusionSteps` | Gets or sets the number of diffusion denoising steps. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumHeads` | Gets or sets the number of attention heads in the denoising network. |
| `NumResBlocks` | Gets or sets the number of U-Net residual blocks per level. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

