---
title: "CenteringMechanism<T>"
description: "Centering mechanism for preventing collapse in self-distillation methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Centering mechanism for preventing collapse in self-distillation methods.

## For Beginners

Centering is a crucial technique in DINO and similar methods
that prevents the teacher network from collapsing to a trivial solution where it
outputs the same constant for all inputs.

## How It Works

**How it works:**

**Why it prevents collapse:** Without centering, the teacher could learn
to output a constant vector for all inputs (trivial solution). By subtracting the
running mean, we ensure the outputs are zero-centered on average, forcing the
network to produce varied outputs.

**Reference:** Caron et al., "Emerging Properties in Self-Supervised Vision
Transformers" (ICCV 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CenteringMechanism(Int32,Double)` | Initializes a new instance of the CenteringMechanism class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `Dimension` | Gets the dimension of the center vector. |
| `Momentum` | Gets the momentum for EMA updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyCenter(Tensor<>)` | Applies centering to the input tensor. |
| `CenterAndUpdate(Tensor<>)` | Applies centering and updates in one step (common usage pattern). |
| `CenterNorm` | Computes the L2 norm of the center (useful for monitoring). |
| `CenterStatistics` | Computes statistics about the center (useful for debugging). |
| `DeepCopy` |  |
| `GetCenter` | Gets the current center values. |
| `GetParameters` |  |
| `Predict(Tensor<>)` |  |
| `Reset` | Resets the center to zeros. |
| `SetCenter([])` | Sets the center values directly. |
| `SetParameters(Vector<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `Update(Tensor<>)` | Updates the center using EMA with the given batch. |
| `UpdateFromMultiple(IList<Tensor<>>)` | Updates the center using multiple batches of outputs. |
| `WithParameters(Vector<>)` |  |

