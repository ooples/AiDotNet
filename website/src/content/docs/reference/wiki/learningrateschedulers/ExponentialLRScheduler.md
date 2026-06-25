---
title: "ExponentialLRScheduler"
description: "Decays the learning rate exponentially every step."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Decays the learning rate exponentially every step.

## For Beginners

This scheduler smoothly reduces the learning rate at every step
by multiplying it by a factor (gamma). Unlike StepLR which makes sudden drops, exponential
decay provides a gradual, continuous reduction. Think of it like gradually releasing pressure
from a gas pedal rather than making sudden brake taps.

## How It Works

ExponentialLR decays the learning rate by gamma every step. This provides a smooth,
continuous decay that can be useful for certain training scenarios.

Formula: lr = base_lr * gamma^step

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExponentialLRScheduler(Double,Double,Double)` | Initializes a new instance of the ExponentialLRScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Gamma` | Gets the multiplicative factor of learning rate decay. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

