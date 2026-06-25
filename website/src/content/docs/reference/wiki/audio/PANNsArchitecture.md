---
title: "PANNsArchitecture"
description: "PANNs architecture variants (legacy enum — the new code uses `PANNsModelOptions` with explicit dim / depth fields)."
section: "API Reference"
---

`Enums` · `AiDotNet.Audio.Fingerprinting`

PANNs architecture variants (legacy enum — the new code uses
`PANNsModelOptions` with explicit dim / depth fields).

## Fields

| Field | Summary |
|:-----|:--------|
| `Cnn10` | CNN10: 10-layer CNN (balanced). |
| `Cnn14` | CNN14: 14-layer CNN (larger, more accurate). |
| `Cnn6` | CNN6: 6-layer CNN (smaller, faster). |
| `MobileNetV2` | MobileNetV2: lightweight mobile variant. |
| `ResNet22` | ResNet-22: residual variant. |
| `ResNet38` | ResNet-38: deeper residual variant. |

