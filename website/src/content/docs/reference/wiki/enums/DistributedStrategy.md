---
title: "DistributedStrategy"
description: "Defines the distributed training strategy to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the distributed training strategy to use.

## For Beginners

These strategies determine how your model and data are split across multiple GPUs or machines.
DDP (Distributed Data Parallel) is the most common and works for 90% of use cases.

## Fields

| Field | Summary |
|:-----|:--------|
| `DDP` | Distributed Data Parallel (DDP) - Most common strategy. |
| `FSDP` | Fully Sharded Data Parallel (FSDP) - PyTorch style, full parameter sharding. |
| `Hybrid` | Hybrid - 3D parallelism combining data + tensor + pipeline. |
| `None` | No distributed training - Single device training (default). |
| `PipelineParallel` | Pipeline Parallel - Model split into stages across GPUs. |
| `TensorParallel` | Tensor Parallel - Individual layers split across GPUs (Megatron-LM style). |
| `ZeRO1` | ZeRO Stage 1 - Only optimizer states are sharded. |
| `ZeRO2` | ZeRO Stage 2 - Optimizer states + gradients are sharded. |
| `ZeRO3` | ZeRO Stage 3 - Full sharding (equivalent to FSDP). |

