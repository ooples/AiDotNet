---
title: "EGLOptions<T>"
description: "Configuration options for Expected Gradient Length strategy."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Strategies`

Configuration options for Expected Gradient Length strategy.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecayFactor` | Gets or sets the decay factor for online accumulation. |
| `Lambda` | Gets or sets the regularization strength (lambda). |
| `MinImportanceValue` | Gets or sets the minimum importance value (to prevent division by zero). |
| `NormalizeImportance` | Gets or sets whether to normalize importance values. |
| `NumSamples` | Gets or sets the number of samples for gradient length estimation. |
| `UseSquaredLength` | Gets or sets whether to use squared gradient lengths. |

