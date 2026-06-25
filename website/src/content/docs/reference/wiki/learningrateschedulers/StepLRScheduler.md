---
title: "StepLRScheduler"
description: "Decays the learning rate by a factor (gamma) every specified number of steps."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Decays the learning rate by a factor (gamma) every specified number of steps.

## For Beginners

This scheduler reduces the learning rate by a fixed amount
at regular intervals. For example, you might reduce the learning rate by 10x every 30 epochs.
This is like slowing down periodically as you get closer to your destination, making
your adjustments more precise as training progresses.

## How It Works

StepLR is one of the simplest and most commonly used learning rate schedulers.
It multiplies the learning rate by gamma every step_size epochs/steps.

Formula: lr = base_lr * gamma^(floor(step / step_size))

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StepLRScheduler(Double,Int32,Double,Double)` | Initializes a new instance of the StepLRScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Gamma` | Gets the multiplicative factor of learning rate decay. |
| `StepSize` | Gets the step size (period of learning rate decay). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

