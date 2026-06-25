---
title: "SimCLR<T>"
description: "SimCLR: A Simple Framework for Contrastive Learning of Visual Representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

## For Beginners

SimCLR is one of the most influential self-supervised learning methods.
It learns representations by maximizing agreement between differently augmented views of the same image
using a contrastive loss.

## How It Works

**How SimCLR works:**

**Key hyperparameters:**

**Reference:** Chen et al., "A Simple Framework for Contrastive Learning of Visual
Representations" (ICML 2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimCLR(INeuralNetwork<>,IProjectorHead<>,SSLConfig)` | Initializes a new instance of the SimCLR class. |

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
| `Create(INeuralNetwork<>,Int32,Int32,Int32)` | Creates a SimCLR instance with default configuration. |

