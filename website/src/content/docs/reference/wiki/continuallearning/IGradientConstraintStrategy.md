---
title: "IGradientConstraintStrategy<T, TInput, TOutput>"
description: "Extended strategy interface for gradient-based constraint strategies."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ContinualLearning.Interfaces`

Extended strategy interface for gradient-based constraint strategies.

## Properties

| Property | Summary |
|:-----|:--------|
| `StoredGradientCount` | Gets the number of tasks with stored gradients. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ProjectGradient(Vector<>)` | Projects a gradient to satisfy all task constraints. |
| `StoreTaskGradient(Vector<>)` | Stores the gradient for a completed task. |
| `ViolatesConstraint(Vector<>)` | Checks if a gradient violates any task constraint. |

