---
title: "LoRAConfiguration"
description: "LoRA configuration for parameter-efficient fine-tuning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

LoRA configuration for parameter-efficient fine-tuning.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the LoRA alpha scaling factor. |
| `BiasMode` | Gets or sets whether to use bias terms in LoRA layers. |
| `Dropout` | Gets or sets the LoRA dropout rate. |
| `QuantizationBits` | Gets or sets the quantization bits when UseQuantization is true. |
| `Rank` | Gets or sets the LoRA rank (r). |
| `TargetModules` | Gets or sets which modules to apply LoRA to. |
| `UseQuantization` | Gets or sets whether to use quantization (QLoRA). |

