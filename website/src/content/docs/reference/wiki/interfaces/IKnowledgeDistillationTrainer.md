---
title: "IKnowledgeDistillationTrainer<T, TInput, TOutput>"
description: "Defines the contract for knowledge distillation trainers that train student models using knowledge transferred from teacher models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for knowledge distillation trainers that train student models
using knowledge transferred from teacher models.

## For Beginners

A knowledge distillation trainer orchestrates the process of
transferring knowledge from a large, accurate teacher model to a smaller, faster student model.

## How It Works

**Why an interface?**

- **Flexibility**: Multiple trainer implementations (standard, self-distillation, multi-teacher, etc.)
- **Testability**: Easy to mock for unit testing
- **Extensibility**: New training strategies can be added without breaking existing code
- **Dependency Injection**: Can be injected into other components

**Common Implementations:**

- **Standard Trainer**: Single teacher → single student
- **Self-Distillation Trainer**: Model teaches itself (improves calibration)
- **Multi-Teacher Trainer**: Multiple teachers → one student (ensemble distillation)
- **Online Trainer**: Teacher updates during student training
- **Mutual Learning Trainer**: Multiple students learn from each other

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationStrategy` | Gets the distillation strategy used for computing loss and gradients. |
| `Teacher` | Gets the teacher model used for distillation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Func<,>,Vector<>,Vector<>)` | Evaluates the student model's accuracy on test data. |
| `Train(Func<,>,Action<>,Vector<>,Vector<>,Int32,Int32,Action<Int32,>)` | Trains the student model for multiple epochs. |
| `TrainBatch(Func<,>,Action<>,Vector<>,Vector<>)` | Trains the student model on a single batch of data. |

