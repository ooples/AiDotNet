---
title: "ReplayBuffer<T>"
description: "A buffer for storing and replaying experiences in reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.ReplayBuffers`

A buffer for storing and replaying experiences in reinforcement learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReplayBuffer` | Initializes a new instance with default settings. |
| `ReplayBuffer(Int32,Nullable<Int32>)` | Initializes a new instance of the ReplayBuffer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the current number of experiences in the buffer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(Experience<>)` | Adds a new experience to the buffer. |
| `Clear` | Clears the buffer. |
| `Sample(Int32)` | Samples a batch of experiences from the buffer. |

