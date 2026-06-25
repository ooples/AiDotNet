---
title: "MomentumEncoder<T>"
description: "Momentum-updated encoder for self-supervised learning methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Momentum-updated encoder for self-supervised learning methods.

## For Beginners

A momentum encoder is a copy of the main encoder that updates
more slowly using exponential moving average (EMA). This provides stable, consistent targets
during self-supervised training.

## How It Works

**Update formula:**

Where m is momentum (typically 0.99-0.9999).

**Why slow updates?**

**Example usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MomentumEncoder(INeuralNetwork<>,Double)` | Initializes a new instance of the MomentumEncoder class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Encoder` |  |
| `Momentum` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CopyFromMainEncoder(INeuralNetwork<>)` |  |
| `Create(,Double,Func<,>)` | Creates a momentum encoder from a main encoder by cloning. |
| `Encode(Tensor<>)` |  |
| `GetParameters` |  |
| `ScheduleMomentum(Double,Double,Int32,Int32)` | Computes the scheduled momentum value based on training progress. |
| `SetMomentum(Double)` |  |
| `SetParameters(Vector<>)` |  |
| `UpdateFromMainEncoder(INeuralNetwork<>)` |  |
| `UpdateFromParameters(Vector<>)` |  |

