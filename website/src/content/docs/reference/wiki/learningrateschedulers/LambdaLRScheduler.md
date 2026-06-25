---
title: "LambdaLRScheduler"
description: "Sets the learning rate using a user-defined lambda function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Sets the learning rate using a user-defined lambda function.

## For Beginners

This scheduler lets you define your own custom learning rate schedule
using a function. The function receives the current step number and returns a value that gets
multiplied with the initial learning rate. For example, returning 0.5 would give half the initial
learning rate. This is useful when you want a schedule that doesn't fit any of the standard patterns.

## How It Works

LambdaLR provides maximum flexibility by allowing you to define any learning rate schedule
as a function of the current step. The lambda function takes the step number and returns
a multiplier that is applied to the base learning rate.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LambdaLRScheduler(Double,Func<Int32,Double>,Double)` | Initializes a new instance of the LambdaLRScheduler class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

