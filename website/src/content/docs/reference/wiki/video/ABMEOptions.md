---
title: "ABMEOptions"
description: "Configuration options for the ABME asymmetric bilateral motion estimation model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Video.Options`

Configuration options for the ABME asymmetric bilateral motion estimation model.

## For Beginners

Most methods assume motion is symmetric (if something moves right
from frame 0, it moves left by the same amount from frame 1). But real motion isn't
symmetric -- a ball speeding up moves more in the second half. ABME estimates motion
independently in both directions, so it handles acceleration, deceleration, and curved
paths much better than symmetric methods.

## How It Works

ABME (Park et al., ICCV 2021) uses asymmetric bilateral motion estimation:

- Bilateral motion estimation: estimates motion from the target time to both input frames

simultaneously (t to 0 and t to 1), rather than from input frames toward the target

- Asymmetric motion model: the two bilateral motion fields are NOT assumed symmetric;

each has its own magnitude and direction, correctly handling non-linear motion paths
(e.g., accelerating objects, curved trajectories)

- Iterative refinement with asymmetric updates: a GRU-based module iteratively refines

both bilateral flows, with separate update heads that can correct each flow independently

- Context-aware synthesis: the final frame is synthesized by combining bilaterally warped

features with a learned blending mask that accounts for occlusion and motion boundaries

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ABMEOptions` | Initializes a new instance with default values. |
| `ABMEOptions(ABMEOptions)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AsymmetricMotion` | Gets or sets whether to use asymmetric motion modeling. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `LearningRate` | Gets or sets the learning rate. |
| `ModelPath` | Gets or sets the path to the ONNX model file. |
| `NumFeatures` | Gets or sets the number of feature channels. |
| `NumPyramidLevels` | Gets or sets the number of pyramid levels for coarse-to-fine estimation. |
| `NumRefinementIters` | Gets or sets the number of GRU refinement iterations for bilateral flow. |
| `NumResBlocks` | Gets or sets the number of residual blocks in the encoder. |
| `OnnxOptions` | Gets or sets the ONNX runtime options. |
| `Variant` | Gets or sets the model variant. |

