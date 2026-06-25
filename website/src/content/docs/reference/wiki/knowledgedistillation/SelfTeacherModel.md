---
title: "SelfTeacherModel<T>"
description: "Self teacher model that uses the student's own predictions from earlier training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Teachers`

Self teacher model that uses the student's own predictions from earlier training.

## For Beginners

Self-distillation is a technique where a model learns from its own
earlier predictions. This teacher stores pre-computed predictions from earlier epochs and
returns them by index via `Int32)`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfTeacherModel(Int32)` | Initializes a new instance of SelfTeacherModel for cached predictions mode. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension of the teacher model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetCachedPrediction(Int32)` | Gets a cached prediction by index. |
| `GetLogits(Vector<>)` | Not supported for `SelfTeacherModel` — always throws. |

