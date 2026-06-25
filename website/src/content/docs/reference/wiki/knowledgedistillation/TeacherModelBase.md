---
title: "TeacherModelBase<TInput, TOutput, T>"
description: "Abstract base class for teacher models used in knowledge distillation."
section: "API Reference"
---

`Base Classes` · `AiDotNet.KnowledgeDistillation`

Abstract base class for teacher models used in knowledge distillation.
Provides common functionality and utilities for teacher model implementations.

## For Beginners

This base class provides common functionality that all teacher models need,
such as numeric operations and input validation. It's a lightweight foundation that derived classes
build upon.

## How It Works

**Why use a base class?**

- **Code Reuse**: Common utilities like numeric operations are available to all implementations
- **Consistency**: All teachers have access to the same helper methods
- **Extensibility**: New teacher types inherit core functionality automatically
- **Maintainability**: Updates to common utilities benefit all implementations

**Architecture Note:** This base class is intentionally minimal. Complex operations like
temperature scaling are handled by distillation strategies, not teachers. Teachers are responsible
only for providing raw logits.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TeacherModelBase` | Initializes the base teacher model and sets up numeric operations. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the number of output dimensions (e.g., number of classes for classification). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyTemperatureSoftmax(,Double)` | Applies temperature-scaled softmax to logits. |
| `GetLogits()` | Gets the teacher's raw logits (pre-softmax outputs) for the given input. |
| `ValidateInput(,String)` | Validates that the input is not null. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for the specified type T. |

