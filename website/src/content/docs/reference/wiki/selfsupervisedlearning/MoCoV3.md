---
title: "MoCoV3<T>"
description: "MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers.

## For Beginners

MoCo v3 adapts momentum contrastive learning specifically for
Vision Transformers (ViT). It simplifies the framework by removing the memory queue and
using in-batch negatives with a symmetric loss.

## How It Works

**Key changes from MoCo v1/v2:**

**Training stability for ViT:**

**Reference:** Chen et al., "An Empirical Study of Training Self-Supervised Vision
Transformers" (ICCV 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MoCoV3(INeuralNetwork<>,IMomentumEncoder<>,IProjectorHead<>,IProjectorHead<>,IProjectorHead<>,SSLConfig)` | Initializes a new instance of the MoCoV3 class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `Name` |  |
| `RequiresMemoryBank` |  |
| `UsesMomentumEncoder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `OnEpochStart(Int32)` |  |

