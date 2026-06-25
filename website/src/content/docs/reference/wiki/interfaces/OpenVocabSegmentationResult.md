---
title: "OpenVocabSegmentationResult<T>"
description: "Result of open-vocabulary segmentation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of open-vocabulary segmentation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassNames` | Class names corresponding to each mask. |
| `Masks` | Per-class masks [numClasses, H, W]. |
| `Scores` | Per-class confidence scores. |
| `SemanticMap` | Per-pixel semantic map [H, W] with class indices. |

