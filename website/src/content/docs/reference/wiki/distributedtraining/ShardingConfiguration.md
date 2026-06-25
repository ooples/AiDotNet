---
title: "ShardingConfiguration<T>"
description: "Default implementation of sharding configuration for distributed training."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.DistributedTraining`

Default implementation of sharding configuration for distributed training.

## For Beginners

This class holds all the settings that control how distributed training works.
You can create an instance with default settings or customize it for your needs.

## How It Works

Example:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShardingConfiguration(ICommunicationBackend<>,Double)` | Creates a new sharding configuration with the specified communication backend. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AutoSyncGradients` |  |
| `CommunicationBackend` |  |
| `EnableGradientCompression` |  |
| `LearningRate` |  |
| `MinimumParameterGroupSize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateDefault(ICommunicationBackend<>)` | Creates a new sharding configuration with default settings and the specified backend. |
| `CreateForHighBandwidth(ICommunicationBackend<>)` | Creates a configuration optimized for high-bandwidth networks (like NVLink between GPUs). |
| `CreateForLowBandwidth(ICommunicationBackend<>)` | Creates a configuration optimized for low-bandwidth networks (like machines connected over ethernet). |

