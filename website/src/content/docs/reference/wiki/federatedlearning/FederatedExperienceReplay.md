---
title: "FederatedExperienceReplay<T>"
description: "Implements Federated Experience Replay for continual learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ContinualLearning`

Implements Federated Experience Replay for continual learning.

## For Beginners

Experience replay is the simplest anti-forgetting technique:
keep a small "memory" of representative examples from old tasks, and mix them in when
training on new data. In federated ER, each client maintains their own replay buffer
locally (no data sharing). When training on task T+1, each client trains on a mix of
new task data and old examples from their buffer.

## How It Works

Algorithm per client:

Reference: Federated Experience Replay for Continual FL (2023).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedExperienceReplay(Int32,Double,Int32)` | Creates a new Federated Experience Replay strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BufferCapacity` | Gets the buffer capacity. |
| `BufferSize` | Gets the current buffer size. |
| `ReplayRatio` | Gets the replay ratio. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddToBuffer([],Int32)` | Adds an example to the replay buffer using reservoir sampling. |
| `AggregateImportance(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ComputeImportance(Vector<>,Matrix<>)` |  |
| `ComputeRegularizationPenalty(Vector<>,Vector<>,Vector<>,Double)` |  |
| `ProjectGradient(Vector<>,Vector<>)` |  |
| `SampleReplay(Int32)` | Samples a batch from the replay buffer. |

