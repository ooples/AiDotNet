# FSDP - Fully Sharded Data Parallel Training

This sample demonstrates how to use FSDP (Fully Sharded Data Parallel) for memory-efficient distributed training of large models with AiDotNet.

## Overview

FSDP enables training models that are too large to fit on a single GPU by:
1. Sharding model parameters across multiple GPUs
2. Gathering parameters on-demand during forward/backward passes
3. Reducing memory footprint per GPU significantly

## Prerequisites

- .NET 8.0 SDK or later
- AiDotNet NuGet package
- Multiple NVIDIA GPUs with CUDA support
- NCCL library for multi-GPU communication

## Running the Sample

```bash
# Single node, multiple GPUs
cd samples/advanced/DistributedTraining/FSDP
dotnet run

# Multiple nodes (set environment variables first)
export MASTER_ADDR=<master-ip>
export MASTER_PORT=29500
export WORLD_SIZE=<total-gpus>
export RANK=<node-rank>
dotnet run
```

## What This Sample Demonstrates

1. **FSDP Configuration**: Setting up sharding strategies
2. **Memory Optimization**: Reducing GPU memory usage
3. **Mixed Precision**: Combining FSDP with FP16/BF16 training
4. **Gradient Checkpointing**: Further memory savings
5. **Model Wrapping**: Applying FSDP to specific layers

## FSDP vs DDP

| Feature | DDP | FSDP |
|---------|-----|------|
| Memory per GPU | Full model | Sharded model |
| Communication | Gradients only | Parameters + Gradients |
| Max model size | ~GPU memory | ~Total memory |
| Overhead | Lower | Higher |
| Best for | Models that fit | Large models |

## Sharding Strategies

### FULL_SHARD (Default)
- Shards parameters, gradients, and optimizer states
- Maximum memory savings
- Higher communication overhead

### SHARD_GRAD_OP
- Only shards gradients and optimizer states
- Moderate memory savings
- Lower communication overhead

### NO_SHARD
- No sharding (similar to DDP)
- Useful for debugging

## Memory Comparison

For a 7B parameter model on 8 GPUs:

| Method | Memory per GPU |
|--------|---------------|
| Single GPU | 28+ GB (OOM) |
| DDP | 28+ GB (OOM) |
| FSDP SHARD_GRAD_OP | ~14 GB |
| FSDP FULL_SHARD | ~8 GB |
| FSDP + Activation Checkpointing | ~5 GB |

## Code Structure

- `Program.cs` - Main entry point with FSDP training loop
- Multi-GPU initialization
- Model wrapping with FSDP
- Training with gradient accumulation
- Checkpointing and loading

## Related Samples

- [DDP](../DDP/) - Basic distributed data parallel
- [ZeRO](../ZeRO/) - DeepSpeed ZeRO optimization
- [Pipeline Parallel](../PipelineParallel/) - Pipeline parallelism

## Learn More

- [Distributed Training Guide](/docs/tutorials/distributed-training/)
- [FSDP Best Practices](/docs/guides/fsdp-best-practices/)
- [AiDotNet.DistributedTraining API Reference](/api/AiDotNet.DistributedTraining/)
