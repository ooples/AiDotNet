---
title: "HyperShotAlgorithm<T, TInput, TOutput>"
description: "Implementation of HyperShot (kernel hypernetwork for few-shot learning)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of HyperShot (kernel hypernetwork for few-shot learning).

## For Beginners

HyperShot learns to create custom distance functions:

**The insight:**
Different tasks need different ways to measure similarity. Comparing dog breeds
requires looking at ear shape and size, while comparing bird species needs beak
and plumage analysis. HyperShot generates the right comparison function for each task.

**How it works:**

1. Extract features from the support set using a shared backbone
2. Feed support features into a hypernetwork (a network that generates another network)
3. The hypernetwork outputs kernel parameters that define how to measure similarity
4. Use the generated kernel to compare query features against support prototypes

**Why a hypernetwork?**
Instead of learning ONE fixed kernel that works for all tasks,
HyperShot generates a CUSTOM kernel for each task, tailored to
what makes the classes in that specific task different.

## How It Works

HyperShot uses a hypernetwork to generate task-specific kernel parameters from the
support set, enabling adaptive similarity computation for few-shot classification.

**Algorithm - HyperShot:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HyperShotAlgorithm(HyperShotOptions<,,>)` | Initializes a new HyperShot meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using the kernel hypernetwork. |
| `GenerateKernelFromSupport(Vector<>)` | Generates task-specific kernel parameters from support set statistics using the hypernetwork. |
| `InitializeHypernetwork` | Initializes the kernel hypernetwork parameters. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_hypernetParams` | Parameters for the kernel hypernetwork. |

