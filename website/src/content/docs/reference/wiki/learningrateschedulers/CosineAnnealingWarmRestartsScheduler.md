---
title: "CosineAnnealingWarmRestartsScheduler"
description: "Sets the learning rate using cosine annealing with warm restarts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Sets the learning rate using cosine annealing with warm restarts.

## For Beginners

Imagine running a race in sprints instead of one continuous run.
After each sprint (cycle), you rest (restart learning rate) and then sprint again. This "warm restart"
approach helps the model escape local minima and often finds better solutions. The sprints can
optionally get longer each time (controlled by T_mult), allowing for more fine-tuning in later cycles.

## How It Works

This scheduler implements the SGDR (Stochastic Gradient Descent with Warm Restarts) algorithm.
It uses cosine annealing but periodically restarts the learning rate to the initial value,
optionally increasing the period between restarts.

Based on the paper "SGDR: Stochastic Gradient Descent with Warm Restarts" by Loshchilov & Hutter.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CosineAnnealingWarmRestartsScheduler(Double,Int32,Int32,Double)` | Initializes a new instance of the CosineAnnealingWarmRestartsScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentCycle` | Gets the current cycle number. |
| `EtaMin` | Gets the minimum learning rate. |
| `T0` | Gets the initial period. |
| `TMult` | Gets the period multiplier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |
| `LoadState(Dictionary<String,Object>)` |  |
| `Reset` |  |
| `Step` |  |

