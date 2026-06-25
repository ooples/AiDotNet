---
title: "GenerativeReplay<T>"
description: "Implements Generative Replay (also known as Deep Generative Replay) for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning`

Implements Generative Replay (also known as Deep Generative Replay) for continual learning.

## For Beginners

Generative Replay uses a generative model (like a VAE or GAN)
to create pseudo-examples from previous tasks instead of storing real examples. This
enables rehearsal without storing actual data, which is useful for privacy-sensitive
applications or when memory is limited.

## How It Works

**How it works:**

**Key Components:**

**Advantages:**

**Reference:** Shin, H., Lee, J.K., Kim, J., and Kim, J. "Continual Learning with
Deep Generative Replay" (2017). NeurIPS.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GenerativeReplay(Int32,Double,Double,Nullable<Int32>)` | Initializes a new instance of the GenerativeReplay class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatesAcrossTasks` |  |
| `ReplayBatchSize` | Gets the replay batch size. |
| `ReplayRatio` | Gets the replay ratio. |
| `TaskCount` | Gets the number of tasks processed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AfterTask(INeuralNetwork<>,ValueTuple<Tensor<>,Tensor<>>,Int32)` |  |
| `BeforeTask(INeuralNetwork<>,Int32)` |  |
| `CombineTensors(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Combines current and replay tensors into a single batch. |
| `ComputeLoss(INeuralNetwork<>)` |  |
| `CreateMixedBatch(Tensor<>,Tensor<>,Int32)` | Creates a mixed training batch combining new data with generated replay data. |
| `GenerateReplaySamples` | Generates pseudo-examples for replay using the generator. |
| `ModifyGradients(INeuralNetwork<>,Vector<>)` |  |
| `Reset` |  |
| `SampleTensor(Tensor<>,Int32[])` | Samples specific indices from a tensor. |
| `SetGenerator(IGenerativeModel<>)` | Sets the generative model used for replay. |

