---
title: "FineTuningStrategy"
description: "Fine-tuning strategies for SSL pretrained encoders."
section: "API Reference"
---

`Enums` · `AiDotNet.SelfSupervisedLearning`

Fine-tuning strategies for SSL pretrained encoders.

## Fields

| Field | Summary |
|:-----|:--------|
| `FullFineTuning` | Update all parameters including encoder. |
| `GradualUnfreezing` | Progressively unfreeze encoder layers. |
| `LinearProbing` | Freeze encoder, only train classifier. |

