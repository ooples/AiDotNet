---
title: "OneCycleLRScheduler"
description: "Implements the 1cycle learning rate policy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Implements the 1cycle learning rate policy.

## For Beginners

The 1cycle policy is like warming up before a workout,
going full intensity during the workout, and then cooling down. The learning rate
starts low (warmup), ramps up to a maximum (peak training), and then decreases
to very low values (fine-tuning). This approach often allows training with higher
maximum learning rates and can achieve better results in fewer epochs.

## How It Works

The 1cycle policy starts with a low learning rate, increases it to a maximum, then
decreases it again. This approach has been shown to enable faster training and
better final performance, especially when combined with momentum cycling.

Based on the paper "Super-Convergence: Very Fast Training of Neural Networks Using
Large Learning Rates" by Leslie N. Smith and Nicholay Topin.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OneCycleLRScheduler(Double,Int32,Double,Double,Double,AnnealingStrategy)` | Initializes a new instance of the OneCycleLRScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxLearningRate` | Gets the maximum learning rate. |
| `PctStart` | Gets the percentage of steps for warmup. |
| `TotalSteps` | Gets the total number of steps. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

