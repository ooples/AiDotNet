---
title: "CAVIAContextInjectionMode"
description: "Specifies how context parameters are injected into the model's computation."
section: "API Reference"
---

`Enums` · `AiDotNet.MetaLearning.Options`

Specifies how context parameters are injected into the model's computation.

## For Beginners

Think of the context as a "task description" that the
model uses to customize its behavior. The injection mode controls where and how
this description is fed into the model.

## How It Works

The injection mode determines how the task-specific context vector is combined
with the model's input or intermediate representations.

## Fields

| Field | Summary |
|:-----|:--------|
| `Addition` | Adds the context vector element-wise to the input features. |
| `Concatenation` | Concatenates the context vector with the input features. |
| `Multiplication` | Multiplies the context vector element-wise with the input features (FiLM-style gating). |

