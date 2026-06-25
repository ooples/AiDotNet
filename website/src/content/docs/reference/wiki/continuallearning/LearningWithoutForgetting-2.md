---
title: "LearningWithoutForgetting<T, TInput, TOutput>"
description: "Learning without Forgetting (LwF) strategy for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Strategies`

Learning without Forgetting (LwF) strategy for continual learning.

## For Beginners

LwF prevents forgetting by using knowledge distillation.
Instead of storing old data, it stores the model's predictions (knowledge) and
trains the new model to match those predictions while also learning the new task.

## How It Works

**How it works:**

**The Math:**

L_distill = T² * KL(softmax(z_teacher/T) || softmax(z_student/T))

Where T is the temperature and z are the logits (pre-softmax outputs).

**Advantages:**

**Disadvantages:**

**Reference:** Li and Hoiem "Learning without Forgetting" (ECCV 2016)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LearningWithoutForgetting(ILossFunction<>,Double,Double)` | Initializes a new LwF strategy with default options. |
| `LearningWithoutForgetting(ILossFunction<>,LwFOptions<>)` | Initializes a new LwF strategy with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationWeight` |  |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `RequiresMemoryBuffer` |  |
| `TeacherModel` |  |
| `Temperature` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustGradients(Vector<>)` |  |
| `AdvanceEpoch` | Advances to the next epoch (for warmup tracking). |
| `ComputeDistillationLoss(,)` |  |
| `ComputeDistillationLoss(Vector<>,Vector<>)` | Computes the distillation loss between teacher and student predictions. |
| `ComputeKLDivergence(Vector<>,Vector<>)` | Computes KL divergence with temperature scaling. |
| `ComputeMSELoss(Vector<>,Vector<>)` | Computes MSE between soft targets. |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `ComputeSoftCrossEntropy(Vector<>,Vector<>)` | Computes soft cross-entropy loss. |
| `ComputeSymmetricKL(Vector<>,Vector<>)` | Computes symmetric KL divergence. |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetDistillationStats` | Gets distillation statistics. |
| `GetEffectiveDistillationWeight` | Gets the effective distillation weight, accounting for warmup. |
| `GetStateForSerialization` |  |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `Reset` |  |
| `SoftmaxWithTemperature(Vector<>,)` | Applies softmax with temperature scaling. |

