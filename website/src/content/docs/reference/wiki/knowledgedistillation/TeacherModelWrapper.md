---
title: "TeacherModelWrapper<T>"
description: "Wraps an existing trained IFullModel to act as a teacher for knowledge distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

Wraps an existing trained IFullModel to act as a teacher for knowledge distillation.

## For Beginners

This class takes any trained IFullModel and adapts it to work
as a teacher in knowledge distillation. The teacher model should already be trained and
perform well on your task.

## How It Works

**Architecture Note:** This is a lightweight adapter that bridges IFullModel
to ITeacherModel. It simply delegates GetLogits() to the underlying model's Predict() method,
since in this architecture, predictions and logits are equivalent.

**Real-world Example:**
Imagine you have a large, accurate neural network trained on your dataset. You can wrap it
with TeacherModelWrapper and use it to train a smaller, faster student model that retains
most of the accuracy but runs much faster.

Common teacher-student scenarios:

- Large neural network (teacher) → Smaller network (student): 40-60% smaller, 95-97% of performance
- Deep network (teacher) → Shallow network (student): 10x faster inference
- Ensemble (teacher) → Single model (student): Deployable on resource-constrained devices

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TeacherModelWrapper(Func<Vector<>,Vector<>>,Int32)` | Initializes a new instance of the TeacherModelWrapper class from a forward function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the number of output dimensions (e.g., number of classes for classification). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLogits(Vector<>)` | Gets the teacher's raw logits (pre-softmax outputs) for the given input. |

