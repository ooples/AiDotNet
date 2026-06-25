---
title: "OptimizationProfileConfig"
description: "Configuration for a single optimization profile (for dynamic shapes)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.TensorRT`

Configuration for a single optimization profile (for dynamic shapes).

## For Beginners

OptimizationProfileConfig provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `InputName` | Gets or sets the input tensor name. |
| `MaxShape` | Gets or sets the maximum shape for this input. |
| `MinShape` | Gets or sets the minimum shape for this input. |
| `OptimalShape` | Gets or sets the optimal shape for this input (used for optimization). |

