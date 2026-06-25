---
title: "ModelShard<T>"
description: "Enables model sharding across multiple devices for large model inference."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Memory`

Enables model sharding across multiple devices for large model inference.

## For Beginners

Large AI models like Stable Diffusion XL may not fit in a single GPU.

Model sharding solves this by:

- Splitting the model layers across devices
- Running each device's layers in sequence
- Passing intermediate results between devices

Example: UNet with 24 layers on 4 GPUs:

- GPU 0: Layers 1-6
- GPU 1: Layers 7-12
- GPU 2: Layers 13-18
- GPU 3: Layers 19-24

Data flows: Input -> GPU0 -> GPU1 -> GPU2 -> GPU3 -> Output

Usage:
```cs
var shard = new ModelShard<float>(layers, numDevices: 4);
var output = shard.Forward(input);
```

## How It Works

Model sharding (also known as model parallelism or pipeline parallelism) splits
a large model across multiple devices (GPUs/CPUs) when it's too large to fit
on a single device.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelShard(IEnumerable<ILayer<>>,Int32,ShardingConfig)` | Initializes model sharding across specified number of devices. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Config` | Sharding configuration. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AssignLayerToDevice(ILayer<>,Int32)` | Assigns a layer to a specific device. |
| `DistributeByMemory(List<ILayer<>>)` | Distributes layers to balance memory usage across devices. |
| `DistributeCustom(List<ILayer<>>)` | Distributes layers using custom device assignments from config. |
| `DistributeEvenly(List<ILayer<>>)` | Distributes layers evenly across devices. |
| `DistributeLayers(List<ILayer<>>)` | Distributes layers across devices using the configured strategy. |
| `EstimateLayerMemory(ILayer<>)` | Estimates memory usage for a layer in bytes. |
| `Forward(Tensor<>)` | Performs forward pass through all sharded layers. |
| `Forward(Tensor<>,Tensor<>)` | Performs forward pass with context (for conditional generation). |
| `GetDeviceLayers(Int32)` | Gets layers assigned to a specific device. |
| `GetDeviceMemoryUsage` | Gets memory usage per device. |
| `GetLayerDevice(ILayer<>)` | Gets the device assignment for a layer. |
| `MoveToDevice(Tensor<>,Int32)` | Moves a tensor to a specific device (placeholder for GPU implementation). |
| `ToString` | Gets a summary of the sharding distribution. |
| `UpdateParameters()` | Updates parameters on all devices. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `_deviceLayers` | Layers assigned to each device. |
| `_deviceMemory` | Memory usage per device in bytes. |
| `_layerDevices` | Device assignment for each layer. |
| `_numDevices` | Number of devices/shards. |
| `_usePipelineParallelism` | Whether to use pipeline parallelism (overlapping compute/transfer). |

