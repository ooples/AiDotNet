---
title: "MobileNetV3WidthMultiplier"
description: "Specifies the width multiplier for MobileNetV3."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the width multiplier for MobileNetV3.

## For Beginners

The width multiplier controls how "wide" the network is.
A multiplier of 1.0 gives the standard network, while 0.75 uses 75% as many channels,
making the network faster but potentially less accurate.

## How It Works

The width multiplier (alpha) scales the number of channels in each layer.
Smaller multipliers result in faster, more compact models at the cost of accuracy.

## Fields

| Field | Summary |
|:-----|:--------|
| `Alpha075` | Width multiplier of 0.75 - Compact model. |
| `Alpha100` | Width multiplier of 1.0 - Standard model. |

