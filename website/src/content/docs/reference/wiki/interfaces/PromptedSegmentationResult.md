---
title: "PromptedSegmentationResult<T>"
description: "Result from a prompted segmentation operation containing one or more mask proposals."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result from a prompted segmentation operation containing one or more mask proposals.

## Properties

| Property | Summary |
|:-----|:--------|
| `LowResLogits` | Low-resolution mask logits that can be fed back as mask prompts for iterative refinement. |
| `Masks` | Predicted binary masks [numMasks, H, W]. |
| `Scores` | Confidence scores for each mask proposal. |
| `StabilityScores` | Stability scores measuring how consistent each mask is under small perturbations. |

