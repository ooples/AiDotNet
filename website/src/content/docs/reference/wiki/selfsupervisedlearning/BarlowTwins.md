---
title: "BarlowTwins<T>"
description: "Barlow Twins: Self-Supervised Learning via Redundancy Reduction."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Barlow Twins: Self-Supervised Learning via Redundancy Reduction.

## For Beginners

Barlow Twins learns representations by making the
cross-correlation matrix between embeddings of two augmented views close to the
identity matrix. This achieves both invariance (diagonal = 1) and reduces
redundancy between features (off-diagonal = 0).

## How It Works

**Key innovations:**

**Loss components:**

**Cross-correlation matrix:**

**Reference:** Zbontar et al., "Barlow Twins: Self-Supervised Learning via
Redundancy Reduction" (ICML 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BarlowTwins(INeuralNetwork<>,IProjectorHead<>,SSLConfig)` | Initializes a new instance of the BarlowTwins class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `Lambda` | Gets the lambda (redundancy reduction weight) used in the loss. |
| `Name` |  |
| `RequiresMemoryBank` |  |
| `UsesMomentumEncoder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(INeuralNetwork<>,Int32,Int32,Int32,Double)` | Creates a Barlow Twins instance with default configuration. |

