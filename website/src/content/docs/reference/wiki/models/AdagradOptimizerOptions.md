---
title: "AdagradOptimizerOptions<T, TInput, TOutput>"
description: "Configuration options for the Adagrad optimization algorithm, which adapts the learning rate for each parameter based on historical gradient information."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Adagrad optimization algorithm, which adapts the learning rate for each parameter based on historical gradient information.

## For Beginners

Adagrad is like a smart teacher that gives more attention to students who need it most.
If a parameter (think of it as a knob that needs adjusting) hasn't changed much in the past, Adagrad will make bigger adjustments to it.
If another parameter has already been adjusted a lot, Adagrad will make smaller changes to it.
This helps the model learn more efficiently, especially when some parameters need more tuning than others.

## How It Works

Adagrad is an optimization algorithm that automatically adjusts the learning rate for each parameter based on how frequently it is updated.
Parameters that are updated more frequently receive smaller learning rates, while parameters that are updated less frequently receive larger learning rates.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the batch size for mini-batch gradient descent. |
| `Epsilon` | Gets or sets a small constant added to the denominator to prevent division by zero. |
| `InitialLearningRate` | Gets or sets the initial learning rate for the Adagrad optimizer, overriding the base class value. |
| `LearningRateDecreaseFactor` | Gets or sets the factor by which the learning rate decreases when performance worsens. |
| `LearningRateIncreaseFactor` | Gets or sets the factor by which the learning rate increases when performance improves. |

