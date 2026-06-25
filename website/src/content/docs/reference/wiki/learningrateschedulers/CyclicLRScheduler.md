---
title: "CyclicLRScheduler"
description: "Implements cyclical learning rate policy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Implements cyclical learning rate policy.

## For Beginners

Instead of always decreasing the learning rate, cyclic learning
rates go up and down in cycles. The idea is that periodically increasing the learning rate
can help the model escape local minima (suboptimal solutions) and explore better solutions.
Think of it like occasionally taking bigger jumps while hiking to avoid getting stuck in small valleys.

## How It Works

CyclicLR cycles the learning rate between two boundaries with a constant frequency.
This approach can help escape local minima and find better solutions by periodically
increasing the learning rate.

Based on the paper "Cyclical Learning Rates for Training Neural Networks" by Leslie N. Smith.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CyclicLRScheduler(Double,Double,Int32,Nullable<Int32>,CyclicMode,Double)` | Initializes a new instance of the CyclicLRScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CycleCount` | Gets the current cycle count. |
| `MaxLearningRate` | Gets the maximum learning rate. |
| `Mode` | Cyclic mode (Triangular / Triangular2 / ExponentialRange). |
| `StepSizeDown` | Gets the step size for decreasing phase. |
| `StepSizeUp` | Gets the step size for increasing phase. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

