---
title: "IMomentumEncoder<T>"
description: "Defines the contract for momentum-updated encoders used in SSL methods."
section: "API Reference"
---

`Interfaces` · `AiDotNet.SelfSupervisedLearning`

Defines the contract for momentum-updated encoders used in SSL methods.

## For Beginners

A momentum encoder is a copy of the main encoder that updates
more slowly using exponential moving average (EMA). This provides stable, consistent targets
during training.

## How It Works

**How it works:**

Where m is typically 0.99-0.999 (very slow updates).

**Why use a momentum encoder?**

**Used by:** MoCo, MoCo v2, MoCo v3, BYOL, DINO

**Not used by:** SimCLR, SimSiam (uses stop-gradient instead), Barlow Twins

## Properties

| Property | Summary |
|:-----|:--------|
| `Encoder` | Gets the underlying momentum-updated encoder network. |
| `Momentum` | Gets the momentum coefficient for EMA updates. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CopyFromMainEncoder(INeuralNetwork<>)` | Copies all parameters from the main encoder (hard copy, not EMA). |
| `Encode(Tensor<>)` | Encodes input using the momentum encoder (no gradient computation). |
| `GetParameters` | Gets all parameters of the momentum encoder. |
| `SetMomentum(Double)` | Sets the momentum coefficient. |
| `SetParameters(Vector<>)` | Sets the parameters of the momentum encoder directly. |
| `UpdateFromMainEncoder(INeuralNetwork<>)` | Updates the momentum encoder parameters using EMA from the main encoder. |
| `UpdateFromParameters(Vector<>)` | Updates the momentum encoder parameters using EMA from parameter vectors. |

