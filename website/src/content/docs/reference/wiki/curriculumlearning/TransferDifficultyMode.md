---
title: "TransferDifficultyMode"
description: "Mode for transfer-based difficulty calculation."
section: "API Reference"
---

`Enums` · `AiDotNet.CurriculumLearning.DifficultyEstimators`

Mode for transfer-based difficulty calculation.

## Fields

| Field | Summary |
|:-----|:--------|
| `Combined` | Combines multiple metrics for robust estimation. |
| `ConfidenceGap` | Uses the confidence gap between models. |
| `LossGap` | Uses the loss gap between teacher and student models. |
| `TeacherLoss` | Uses only the teacher model's loss as difficulty. |

