---
title: "BYOL<T>"
description: "BYOL: Bootstrap Your Own Latent - Self-supervised learning without negative samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

BYOL: Bootstrap Your Own Latent - Self-supervised learning without negative samples.

## For Beginners

BYOL is a breakthrough method that learns representations
without requiring negative samples. It uses an online network that learns to predict
the output of a target network, which is updated as an exponential moving average (EMA)
of the online network.

## How It Works

**Key innovations:**

**Architecture:**

**Why it doesn't collapse:** The combination of the predictor (asymmetry),
EMA updates (target moves slowly), and batch normalization prevents trivial solutions.

**Reference:** Grill et al., "Bootstrap Your Own Latent - A New Approach to
Self-Supervised Learning" (NeurIPS 2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BYOL(INeuralNetwork<>,IMomentumEncoder<>,SymmetricProjector<>,SymmetricProjector<>,SSLConfig)` | Initializes a new instance of the BYOL class. |

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
| `Create(INeuralNetwork<>,Func<INeuralNetwork<>,INeuralNetwork<>>,Int32,Int32,Int32)` | Creates a BYOL instance with default configuration. |
| `OnEpochStart(Int32)` |  |

