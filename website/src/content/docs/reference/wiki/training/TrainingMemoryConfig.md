---
title: "TrainingMemoryConfig"
description: "Configuration for training memory management including gradient checkpointing, activation pooling, and model sharding."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Training.Memory`

Configuration for training memory management including gradient checkpointing,
activation pooling, and model sharding.

## For Beginners

Training neural networks requires a lot of memory:

1. **Model weights**: The parameters being trained
2. **Activations**: Intermediate results saved for backpropagation
3. **Gradients**: Computed during backward pass

This configuration helps reduce memory usage through:

- **Gradient Checkpointing**: Trade compute for memory by recomputing activations
- **Activation Pooling**: Reuse memory buffers to reduce garbage collection
- **Model Sharding**: Split large models across multiple GPUs

## How It Works

Example usage with AiModelBuilder:

## Properties

| Property | Summary |
|:-----|:--------|
| `CheckpointAttentionLayers` | Gets or sets whether to checkpoint attention layers specifically. |
| `CheckpointEveryNLayers` | Gets or sets how often to create checkpoints (every N layers). |
| `CheckpointResidualBlocks` | Gets or sets whether to checkpoint residual blocks. |
| `DefaultInitializationStrategy` | Gets or sets the default initialization strategy type. |
| `HighActivationMemoryThresholdBytes` | Gets or sets the activation memory threshold (in bytes) above which a layer is automatically checkpointed regardless of other settings. |
| `MaxPoolMemoryMB` | Gets or sets the maximum memory for the activation pool in megabytes. |
| `MemoryWarningThreshold` | Gets or sets the warning threshold for memory usage (as fraction of max). |
| `MicroBatchCount` | Gets or sets the number of micro-batches for pipeline parallelism. |
| `NumDevices` | Gets or sets the number of devices for model sharding. |
| `ShardingStrategy` | Gets or sets the sharding strategy. |
| `TensorPoolOptions` | Gets or sets the tensor pooling options. |
| `TrackMemoryStatistics` | Gets or sets whether to track detailed memory statistics. |
| `UseActivationPooling` | Gets or sets whether to use activation pooling. |
| `UseGradientCheckpointing` | Gets or sets whether to use gradient checkpointing. |
| `UsePipelineParallelism` | Gets or sets whether to use pipeline parallelism. |
| `UseSharedTensorPool` | Gets or sets whether to use a shared global tensor pool. |
| `UseTensorPooling` | Gets or sets whether tensor pooling is enabled. |
| `WeightsFileFormat` | Gets or sets the format of the weights file. |
| `WeightsFilePath` | Gets or sets the path to a weights file for FromFile initialization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggressivePooling(Int32)` | Creates a configuration with aggressive memory pooling. |
| `Default` | Creates a default configuration with minimal memory optimization. |
| `FastConstruction` | Creates a configuration for fast model construction (good for testing). |
| `ForConvNets` | Creates a configuration optimized for convolutional networks. |
| `ForTransferLearning(String,WeightFileFormat)` | Creates a configuration for transfer learning from a pre-trained model. |
| `ForTransformers` | Creates a configuration optimized for transformer models. |
| `MemoryEfficient` | Creates a memory-efficient configuration for large models. |
| `MultiGpu(Int32)` | Creates a configuration for multi-GPU training. |
| `SpeedOptimized` | Creates a speed-optimized configuration (less memory optimization). |

