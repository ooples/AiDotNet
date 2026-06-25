---
title: "SimSiam<T>"
description: "SimSiam: Exploring Simple Siamese Representation Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

SimSiam: Exploring Simple Siamese Representation Learning.

## For Beginners

SimSiam shows that simple Siamese networks can learn
meaningful representations without negative pairs, momentum encoder, or large batches.
The key is the stop-gradient operation applied to one branch.

## How It Works

**Key innovations:**

**Architecture:**

**Why it works:** The stop-gradient prevents both branches from collapsing
to the same constant output. The predictor makes one branch "predict" the other,
creating useful gradients for learning.

**Reference:** Chen and He, "Exploring Simple Siamese Representation Learning"
(CVPR 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SimSiam(INeuralNetwork<>,SymmetricProjector<>,SSLConfig)` | Initializes a new instance of the SimSiam class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `Name` |  |
| `RequiresMemoryBank` |  |
| `SymmetricProjector` | Gets the typed symmetric projector. |
| `UsesMomentumEncoder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(INeuralNetwork<>,Int32,Int32,Int32,Int32)` | Creates a SimSiam instance with default configuration. |

