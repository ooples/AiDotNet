---
title: "SSLStepResult<T>"
description: "Result of a single SSL training step."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Result of a single SSL training step.

## For Beginners

This contains all the information from one training iteration,
including the loss value and any additional metrics specific to the SSL method.

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentLearningRate` | Gets or sets the current learning rate (if adaptive). |
| `CurrentTemperature` | Gets or sets the current temperature parameter (if applicable). |
| `Loss` | Gets or sets the primary loss value for this step. |
| `Metrics` | Gets or sets additional metrics specific to the SSL method. |
| `NumNegativePairs` | Gets or sets the number of negative pairs in this batch. |
| `NumPositivePairs` | Gets or sets the number of positive pairs in this batch. |

