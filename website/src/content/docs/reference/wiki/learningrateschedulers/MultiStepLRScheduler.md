---
title: "MultiStepLRScheduler"
description: "Decays the learning rate by gamma at each milestone step."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Decays the learning rate by gamma at each milestone step.

## For Beginners

Unlike StepLR which decays at regular intervals, MultiStepLR lets you
specify exactly which steps to decay the learning rate at. For example, you might want to decay
at epochs 30, 60, and 90, rather than every 30 epochs. This gives you more control over the training schedule.

## How It Works

MultiStepLR decays the learning rate by gamma once the number of steps reaches one of the milestones.
This allows for non-uniform decay schedules where you specify exactly when the learning rate should decrease.

This is useful when you know from experience or experimentation that certain epochs are good
points to reduce the learning rate.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiStepLRScheduler(Double,Int32[],Double,Double)` | Initializes a new instance of the MultiStepLRScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Gamma` | Gets the multiplicative factor of learning rate decay. |
| `Milestones` | Gets the milestones. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

