---
title: "DetachedTensor<T>"
description: "A wrapper that marks a tensor as detached from the computation graph."
section: "API Reference"
---

`Structs` · `AiDotNet.SelfSupervisedLearning`

A wrapper that marks a tensor as detached from the computation graph.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DetachedTensor(Tensor<>)` | Initializes a new DetachedTensor wrapping the given tensor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Data` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `From(Tensor<>)` | Creates a DetachedTensor from a regular tensor. |
| `op_Implicit(DetachedTensor<>)~Tensor<>` | Implicitly converts a DetachedTensor to its underlying Tensor. |

