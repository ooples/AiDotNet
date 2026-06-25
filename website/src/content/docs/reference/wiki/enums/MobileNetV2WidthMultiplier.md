---
title: "MobileNetV2WidthMultiplier"
description: "Specifies the width multiplier for MobileNetV2."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the width multiplier for MobileNetV2.

## For Beginners

The width multiplier controls how "wide" the network is.
A multiplier of 1.0 gives the standard network, while 0.5 uses half as many channels,
making the network faster but potentially less accurate.

## How It Works

The width multiplier (alpha) scales the number of channels in each layer.
Smaller multipliers result in faster, more compact models at the cost of accuracy.

## Fields

| Field | Summary |
|:-----|:--------|
| `Alpha035` | Width multiplier of 0.35 - Extremely compact model. |
| `Alpha050` | Width multiplier of 0.5 - Very compact model. |
| `Alpha075` | Width multiplier of 0.75 - Compact model. |
| `Alpha100` | Width multiplier of 1.0 - Standard model. |
| `Alpha125` | Width multiplier of 1.25 - Wider model for improved accuracy. |
| `Alpha130` | Width multiplier of 1.3 - Wider model for higher accuracy. |
| `Alpha140` | Width multiplier of 1.4 - Even wider model for maximum accuracy. |

