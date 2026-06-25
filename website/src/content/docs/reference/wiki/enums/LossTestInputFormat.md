---
title: "LossTestInputFormat"
description: "Describes what kind of test data a loss function expects."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Describes what kind of test data a loss function expects.
The test scaffold generator uses this to create appropriate test vectors.

## Fields

| Field | Summary |
|:-----|:--------|
| `Continuous` | Standard continuous values in [0, 1] range. |
| `CriticScores` | Wasserstein critic scores with signed labels. |
| `MarginBased` | Margin-based with predictions in [0, 1] and binary labels. |
| `OrdinalCategories` | Ordinal category indices. |
| `ProbabilityDistribution` | Probability distribution where values are in [0, 1]. |
| `RawLogits` | Raw logits (unbounded real values) with one-hot or soft targets. |
| `SegmentationMask` | Segmentation masks in [0, 1] range. |
| `SignedLabels` | Signed labels in {-1, +1} with predictions as real-valued scores. |
| `SimilarityLabels` | Binary similarity labels in {0, 1} with distance-based predictions. |

