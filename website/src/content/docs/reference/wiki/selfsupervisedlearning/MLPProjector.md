---
title: "MLPProjector<T>"
description: "Multi-layer perceptron (MLP) projection head for self-supervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Multi-layer perceptron (MLP) projection head for self-supervised learning.

## For Beginners

An MLP projector transforms encoder outputs into a lower-dimensional
space optimized for the SSL loss. This is the standard projector used in SimCLR, MoCo v2, BYOL.

## How It Works

**Architecture:**

**Why MLP over Linear?**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MLPProjector(Int32,Int32,Int32,Boolean,Nullable<Int32>)` | Initializes a new instance of the MLPProjector class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vector operations and GPU/CPU acceleration. |
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

