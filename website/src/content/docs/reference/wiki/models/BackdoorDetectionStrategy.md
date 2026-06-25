---
title: "BackdoorDetectionStrategy"
description: "Specifies the backdoor detection strategy for federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the backdoor detection strategy for federated learning.

## Fields

| Field | Summary |
|:-----|:--------|
| `DirectionAlignmentInspector` | Direction Alignment Inspector — detects anomalous gradient directions per subspace. |
| `NeuralCleanse` | Neural Cleanse — reverse-engineers potential triggers via L1-norm outlier detection. |
| `None` | No backdoor defense — standard aggregation without detection. |

