---
title: "MASImportanceMode"
description: "Mode for computing parameter importance in MAS."
section: "API Reference"
---

`Enums` · `AiDotNet.ContinualLearning.Strategies`

Mode for computing parameter importance in MAS.

## Fields

| Field | Summary |
|:-----|:--------|
| `FisherDiagonal` | Use diagonal of Fisher Information (hybrid with EWC). |
| `Hebbian` | Hebbian-style importance based on activation magnitudes. |
| `OutputSensitivity` | Original MAS: use gradient of output L2 norm. |
| `RandomProjection` | Use gradient of output with respect to randomly sampled directions. |

