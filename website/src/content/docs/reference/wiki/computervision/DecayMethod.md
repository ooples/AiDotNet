---
title: "DecayMethod<T>"
description: "Soft-NMS decay method."
section: "API Reference"
---

`Enums` · `AiDotNet.ComputerVision.Detection.PostProcessing`

Soft-NMS decay method.

## Fields

| Field | Summary |
|:-----|:--------|
| `Gaussian` | Gaussian decay: score = score * exp(-IoU^2 / sigma). |
| `Linear` | Linear decay: score = score * (1 - IoU). |

