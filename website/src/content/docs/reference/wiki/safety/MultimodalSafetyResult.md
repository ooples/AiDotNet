---
title: "MultimodalSafetyResult"
description: "Detailed result from multimodal safety evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Multimodal`

Detailed result from multimodal safety evaluation.

## For Beginners

MultimodalSafetyResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `AlignmentScore` | Cross-modal alignment score (0.0 = misaligned, 1.0 = aligned). |
| `CrossModalAttackDetected` | Whether a cross-modal attack was detected. |
| `IsSafe` | Whether the multimodal content is safe overall. |
| `Modalities` | Modalities involved in the evaluation. |

