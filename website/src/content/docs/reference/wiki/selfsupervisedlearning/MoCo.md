---
title: "MoCo<T>"
description: "MoCo: Momentum Contrast for Unsupervised Visual Representation Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.

## For Beginners

MoCo is a contrastive learning method that uses a momentum encoder
and a memory queue to provide a large pool of consistent negative samples without requiring
huge batch sizes.

## How It Works

**Key innovations:**

**How MoCo works:**

**Reference:** He et al., "Momentum Contrast for Unsupervised Visual Representation
Learning" (CVPR 2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MoCo(INeuralNetwork<>,IMomentumEncoder<>,IProjectorHead<>,IProjectorHead<>,Int32,SSLConfig)` | Initializes a new instance of the MoCo class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MemoryBank` | Gets the memory bank used for negative samples. |
| `Name` |  |
| `RequiresMemoryBank` |  |
| `UsesMomentumEncoder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAdditionalParameterCount` |  |
| `GetAdditionalParameters` |  |
| `Reset` |  |

