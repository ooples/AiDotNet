---
title: "VisionScanPattern"
description: "Defines the scan pattern used by the Vision Mamba model to convert 2D patch grids into 1D sequences."
section: "API Reference"
---

`Enums` · `AiDotNet.NeuralNetworks`

Defines the scan pattern used by the Vision Mamba model to convert 2D patch grids into 1D sequences.

## Fields

| Field | Summary |
|:-----|:--------|
| `Bidirectional` | Bidirectional scan: forward + reverse, used by the original Vision Mamba (Vim) paper. |
| `Continuous` | Continuous/zigzag scan preserving spatial locality, used by PlainMamba. |
| `CrossScan` | Cross-scan: four directional scans (L→R, R→L, T→B, B→T), used by VMamba. |

