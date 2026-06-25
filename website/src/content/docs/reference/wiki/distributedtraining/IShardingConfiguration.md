---
title: "IShardingConfiguration<T>"
description: "Configuration for parameter sharding in distributed training."
section: "API Reference"
---

`Interfaces` · `AiDotNet.DistributedTraining`

Configuration for parameter sharding in distributed training.

## For Beginners

This configuration tells the sharding system how to divide up parameters
and how to handle communication. Think of it as the "rules" for how the
team collaborates.

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoSyncGradients` | Gets whether to automatically synchronize gradients after backward pass. |
| `CommunicationBackend` | Gets the communication backend to use for distributed operations. |
| `EnableGradientCompression` | Gets whether to enable gradient compression to reduce communication costs. |
| `LearningRate` | Gets the learning rate for gradient application during training. |
| `MinimumParameterGroupSize` | Gets the minimum parameter group size for sharding. |

