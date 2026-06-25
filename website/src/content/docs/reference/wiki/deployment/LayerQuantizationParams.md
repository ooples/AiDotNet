---
title: "LayerQuantizationParams"
description: "Per-layer quantization parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Optimization.Quantization`

Per-layer quantization parameters.

## For Beginners

LayerQuantizationParams provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `BitWidth` | Gets or sets the bit width for this layer (if different from global). |
| `Mode` | Gets or sets custom quantization mode for this layer. |
| `ScaleFactor` | Gets or sets the scale factor for this layer. |
| `Skip` | Gets or sets whether to skip quantization for this layer. |
| `ZeroPoint` | Gets or sets the zero point for this layer. |

