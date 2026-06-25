---
title: "MobileNetSearchSpace<T>"
description: "Defines the MobileNet-based search space for neural architecture search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.SearchSpace`

Defines the MobileNet-based search space for neural architecture search.
Includes inverted residual blocks, depthwise separable convolutions, and squeeze-excitation.

## Properties

| Property | Summary |
|:-----|:--------|
| `DepthMultiplier` | Depth multiplier for scaling network depth |
| `ExpansionRatios` | Expansion ratios for inverted residual blocks |
| `KernelSizes` | Kernel sizes to search over |
| `WidthMultiplier` | Width multiplier for scaling channel counts |

