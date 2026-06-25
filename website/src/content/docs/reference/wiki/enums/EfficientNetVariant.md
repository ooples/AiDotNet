---
title: "EfficientNetVariant"
description: "Specifies the EfficientNet model variant."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the EfficientNet model variant.

## For Beginners

The variant number (B0-B7) indicates the scale of the network.
B0 is the smallest and fastest, while B7 is the largest with the highest accuracy.
Choose based on your accuracy requirements and computational budget.

## How It Works

EfficientNet variants use compound scaling to balance network depth, width, and resolution.
Each variant (B0-B7) represents a different scale factor, with larger variants offering
better accuracy at the cost of more computation.

## Fields

| Field | Summary |
|:-----|:--------|
| `B0` | EfficientNet-B0: Base model (5.3M parameters, 224x224 input). |
| `B1` | EfficientNet-B1: Scaled model (7.8M parameters, 240x240 input). |
| `B2` | EfficientNet-B2: Scaled model (9.2M parameters, 260x260 input). |
| `B3` | EfficientNet-B3: Scaled model (12M parameters, 300x300 input). |
| `B4` | EfficientNet-B4: Scaled model (19M parameters, 380x380 input). |
| `B5` | EfficientNet-B5: Scaled model (30M parameters, 456x456 input). |
| `B6` | EfficientNet-B6: Scaled model (43M parameters, 528x528 input). |
| `B7` | EfficientNet-B7: Largest model (66M parameters, 600x600 input). |
| `Custom` | Custom EfficientNet variant for testing with minimal layers. |

