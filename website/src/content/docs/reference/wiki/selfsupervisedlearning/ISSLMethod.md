---
title: "ISSLMethod<T>"
description: "Defines the contract for self-supervised learning methods."
section: "API Reference"
---

`Interfaces` · `AiDotNet.SelfSupervisedLearning`

Defines the contract for self-supervised learning methods.

## For Beginners

Self-supervised learning methods learn useful representations from
unlabeled data. They create "pretext tasks" that provide supervision signals without human labels.

## How It Works

Each SSL method implements this interface and provides:

**Example usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` | Gets the category of this SSL method. |
| `Name` | Gets the name of this SSL method. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `RequiresMemoryBank` | Indicates whether this method requires a memory bank for negative samples. |
| `UsesMomentumEncoder` | Indicates whether this method uses a momentum-updated encoder. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Encode(Tensor<>)` | Encodes input data into learned representations. |
| `GetEncoder` | Gets the underlying encoder neural network. |
| `GetParameters` | Gets the current parameters of the SSL method for serialization. |
| `OnEpochEnd(Int32)` | Called at the end of each training epoch. |
| `OnEpochStart(Int32)` | Called at the start of each training epoch. |
| `Reset` | Resets the SSL method to its initial state. |
| `SetParameters(Vector<>)` | Sets the parameters of the SSL method from a serialized vector. |
| `TrainStep(Tensor<>,SSLAugmentationContext<>)` | Performs a single training step on a batch of data. |

