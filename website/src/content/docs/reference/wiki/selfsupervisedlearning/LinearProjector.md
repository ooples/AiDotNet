---
title: "LinearProjector<T>"
description: "Linear projection head for self-supervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Linear projection head for self-supervised learning.

## For Beginners

A linear projector is the simplest projection head - just a single
linear transformation (matrix multiplication + bias). While simpler than MLP projectors,
linear projectors can still be effective in some scenarios.

## How It Works

**Architecture:**

**When to use Linear vs MLP:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearProjector(Int32,Int32,Boolean,Nullable<Int32>)` | Initializes a new instance of the LinearProjector class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDimension` |  |
| `InputDimension` |  |
| `OutputDimension` |  |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `Project(Tensor<>)` |  |
| `Reset` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` |  |

