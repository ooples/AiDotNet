---
title: "LearningWithoutForgetting<T>"
description: "Implements Learning without Forgetting (LwF) for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Learning without Forgetting (LwF) for continual learning.

## For Beginners

Learning without Forgetting is like teaching a student to solve
new problems while making sure they remember how to solve old ones. It does this by asking
the student to match their old answers (before learning new material) even as they learn
new things.

## How It Works

**How it works:**

**Reference:** Li and Hoiem, "Learning without Forgetting" (2017). IEEE TPAMI.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LearningWithoutForgetting(Double,Double)` | Initializes a new instance of the LearningWithoutForgetting class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `TaskCount` | Gets the number of tasks that have stored predictions. |
| `Temperature` | Gets or sets the temperature for knowledge distillation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `ComputeDistillationLoss(Tensor<>,Tensor<>)` | Computes the distillation loss between current and old predictions. |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `PrepareDistillation(INeuralNetwork<>,Tensor<>,Int32)` | Prepares distillation by recording the old model's predictions on new task inputs. |
| `Reset` |  |
| `TemperatureSoftmax(Tensor<>,Int32,Int32)` | Applies temperature-scaled softmax to logits for a single sample. |

