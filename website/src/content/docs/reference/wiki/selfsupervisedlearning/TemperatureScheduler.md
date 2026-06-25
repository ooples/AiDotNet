---
title: "TemperatureScheduler"
description: "Schedules temperature parameters during self-supervised learning training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Schedules temperature parameters during self-supervised learning training.

## For Beginners

Temperature controls how "sharp" or "soft" the probability
distribution is in contrastive learning. Different methods schedule temperature differently
during training to improve learning dynamics.

## How It Works

**Temperature effects:**

**Common scheduling strategies:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TemperatureScheduler(TemperatureScheduleType,Double,Double,Int32,Int32)` | Initializes a new instance of the TemperatureScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FinalTemperature` | Gets the final temperature value. |
| `InitialTemperature` | Gets the initial temperature value. |
| `ScheduleType` | Gets the schedule type being used. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Constant(Double)` | Creates a scheduler for constant temperature (SimCLR, MoCo default). |
| `CosineAnneal(Double,Double,Int32)` | Creates a scheduler with cosine annealing from high to low temperature. |
| `ForDINOTeacher(Double,Double,Int32,Int32)` | Creates a scheduler for DINO teacher temperature warmup. |
| `GetTemperature(Int32)` | Gets the temperature value for the current training step. |
| `GetTemperatureForEpoch(Int32,Int32)` | Gets the temperature value for the current epoch. |

