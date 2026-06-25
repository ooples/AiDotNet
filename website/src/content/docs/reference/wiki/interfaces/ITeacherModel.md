---
title: "ITeacherModel<TInput, TOutput>"
description: "Represents a trained teacher model for knowledge distillation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a trained teacher model for knowledge distillation.

## For Beginners

Knowledge distillation is a technique where a smaller "student" model
learns from a larger "teacher" model. The teacher model acts as a guide, sharing not just the correct
answers but also its "knowledge" about the relationships between different classes.

## How It Works

Think of it like a master teaching an apprentice. The teacher doesn't just tell the apprentice
the final answer, but shares their reasoning and understanding, which helps the apprentice learn more effectively.

**Architecture Note:** This interface defines a lightweight contract for teacher models
in knowledge distillation. Teachers are inference-only - they provide predictions but don't need
training capabilities. For wrapping a trained IFullModel as a teacher, use TeacherModelWrapper.

**Design Principles:**

- Teachers are frozen/pre-trained - no training methods needed
- Temperature scaling handled by distillation strategy, not teacher
- Focuses on core functionality: get predictions and report output dimension
- Avoids type-unsafe methods (no object? returns)

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the number of output dimensions (e.g., number of classes for classification). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLogits()` | Gets the teacher's raw logits (pre-softmax outputs) for the given input. |

