---
title: "MoCoV2<T>"
description: "MoCo v2: Improved Baselines with Momentum Contrastive Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

MoCo v2: Improved Baselines with Momentum Contrastive Learning.

## For Beginners

MoCo v2 improves on MoCo v1 by incorporating ideas from SimCLR:
an MLP projection head and stronger augmentations. This combines MoCo's memory efficiency
with SimCLR's representation quality improvements.

## How It Works

**Key improvements over MoCo v1:**

**Result:** MoCo v2 matches SimCLR performance with much smaller batch sizes
(256 vs 4096-8192).

**Reference:** Chen et al., "Improved Baselines with Momentum Contrastive Learning"
(arXiv 2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MoCoV2(INeuralNetwork<>,IMomentumEncoder<>,IProjectorHead<>,IProjectorHead<>,Int32,SSLConfig)` | Initializes a new instance of the MoCoV2 class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(INeuralNetwork<>,Func<INeuralNetwork<>,INeuralNetwork<>>,Int32,Int32,Int32,Int32)` | Creates a MoCo v2 instance with default configuration. |

