---
title: "CurriculumTeacherModel<T>"
description: "Curriculum teacher that wraps a base teacher for curriculum learning scenarios."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Teachers`

Curriculum teacher that wraps a base teacher for curriculum learning scenarios.

## How It Works

**Architecture Note:** This class provides a simple wrapper around a base teacher.
Curriculum learning logic (adjusting difficulty over time) should be implemented in the
training loop or distillation strategy, not in the teacher model.

The teacher model's responsibility is only to provide predictions (logits).
Curriculum decisions (which samples to show, how to adjust temperature/alpha) belong
in the strategy or trainer layer.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumTeacherModel(ITeacherModel<Vector<>,Vector<>>)` | Initializes a new instance of the CurriculumTeacherModel class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension from the base teacher. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLogits(Vector<>)` | Gets logits from the base teacher. |

