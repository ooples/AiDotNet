---
title: "DistributedExtensions"
description: "Provides extension methods for easily enabling distributed training on models and optimizers."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.DistributedTraining`

Provides extension methods for easily enabling distributed training on models and optimizers.

## For Beginners

Extension methods allow you to add new functionality to existing classes without modifying them.
These extensions add a simple .AsDistributed() method to models and optimizers,
making it incredibly easy to enable distributed training with just one line of code!

## How It Works

Example:

## Methods

| Method | Summary |
|:-----|:--------|
| `AsDistributed(IFullModel<,,>,ICommunicationBackend<>)` | Wraps a model with distributed training capabilities using the specified communication backend. |
| `AsDistributed(IFullModel<,,>,IShardingConfiguration<>)` | Wraps a model with distributed training capabilities using a custom configuration. |
| `AsDistributed(IOptimizer<,,>,ICommunicationBackend<>)` | Wraps an optimizer with distributed training capabilities. |
| `AsDistributed(IOptimizer<,,>,IShardingConfiguration<>)` | Wraps an optimizer with distributed training capabilities using a custom configuration. |
| `AsDistributedForHighBandwidth(IFullModel<,,>,ICommunicationBackend<>)` | Creates a distributed model with optimized settings for high-bandwidth networks. |
| `AsDistributedForLowBandwidth(IFullModel<,,>,ICommunicationBackend<>)` | Creates a distributed model with optimized settings for low-bandwidth networks. |

