---
title: "SSLDistributedConfig"
description: "Configuration for distributed SSL training using DDP (Distributed Data Parallel)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SelfSupervisedLearning`

Configuration for distributed SSL training using DDP (Distributed Data Parallel).

## For Beginners

This configuration enables training across multiple GPUs or machines.
DDP (Distributed Data Parallel) is the industry-standard approach used by PyTorch, TensorFlow,
and JAX for distributed training.

## How It Works

**How DDP works for SSL:**

**SSL-specific benefits:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Backend` | Gets or sets the communication backend type. |
| `Enabled` | Gets or sets whether distributed training is enabled. |
| `FindUnusedParameters` | Gets or sets whether to use find_unused_parameters behavior. |
| `GradientSyncFrequency` | Gets or sets the gradient synchronization frequency. |
| `Rank` | Gets or sets the rank of this worker (0-indexed). |
| `SharedMemoryQueue` | Gets or sets whether all workers share the same memory queue (for MoCo). |
| `SyncBatchNorm` | Gets or sets whether to synchronize BatchNorm statistics across workers. |
| `UseGradientCompression` | Gets or sets whether to use gradient compression for communication. |
| `WorldSize` | Gets or sets the number of workers (GPUs/processes) for distributed training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

