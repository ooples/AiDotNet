---
title: "CurriculumStrategy"
description: "Defines the curriculum learning strategy direction."
section: "API Reference"
---

`Enums` · `AiDotNet.KnowledgeDistillation.Teachers`

Defines the curriculum learning strategy direction.

## How It Works

Note: This enum is maintained for backward compatibility. Curriculum logic
should be implemented in custom distillation strategies or training loops.

## Fields

| Field | Summary |
|:-----|:--------|
| `EasyToHard` | Start with easy examples and gradually increase difficulty. |
| `HardToEasy` | Start with hard examples and gradually decrease difficulty. |

