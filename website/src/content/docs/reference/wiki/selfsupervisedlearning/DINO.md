---
title: "DINO<T>"
description: "DINO: Self-Distillation with No Labels - a self-supervised method for Vision Transformers."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

DINO: Self-Distillation with No Labels - a self-supervised method for Vision Transformers.

## For Beginners

DINO is a self-supervised method specifically designed for
Vision Transformers (ViT). It learns by having a student network predict the output
of a teacher network, where the teacher is an EMA of the student.

## How It Works

**Key innovations:**

**Architecture:**

**Reference:** Caron et al., "Emerging Properties in Self-Supervised Vision
Transformers" (ICCV 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DINO(INeuralNetwork<>,IMomentumEncoder<>,IProjectorHead<>,IProjectorHead<>,Int32,SSLConfig)` | Initializes a new instance of the DINO class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(INeuralNetwork<>,Func<INeuralNetwork<>,INeuralNetwork<>>,Int32,Int32,Int32,Int32)` | Creates a DINO instance with default configuration. |

