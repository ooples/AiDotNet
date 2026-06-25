---
title: "AdaptiveTeacherModel<T>"
description: "Adaptive teacher model that wraps a base teacher and provides its logits."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Teachers`

Adaptive teacher model that wraps a base teacher and provides its logits.

## How It Works

**Architecture Note:** This class has been simplified to match the current architecture
where temperature scaling is handled by distillation strategies, not teachers. The adaptive
features (dynamic temperature adjustment based on student performance) have been removed as they
belong in the strategy layer.

For adaptive temperature scaling, implement a custom IDistillationStrategy that monitors
student performance and adjusts temperature accordingly.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdaptiveTeacherModel(ITeacherModel<Vector<>,Vector<>>)` | Initializes a new instance of the AdaptiveTeacherModel class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLogits(Vector<>)` | Gets logits from the base teacher. |

