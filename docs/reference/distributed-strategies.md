---
layout: default
title: Distributed Strategies
parent: Reference
nav_order: 6
permalink: /reference/distributed-strategies/
---

# Distributed Training Strategies
{: .no_toc }

Complete reference for all 10+ distributed training strategies in AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AiDotNet supports multiple distributed training strategies to scale training across multiple GPUs and nodes:

| Strategy | Memory per GPU | Communication | Best For |
|:---------|:---------------|:--------------|:---------|
| DDP | Full model | Gradient sync | Models that fit in GPU memory |
| FSDP | Sharded model | All-gather | Large models |
| ZeRO-1 | Sharded optimizer | Optimizer sync | Medium models |
| ZeRO-2 | + Sharded gradients | Gradient sync | Large models |
| ZeRO-3 | + Sharded params | All-gather | Very large models |
| Pipeline | Split by layers | Forward/backward | Deep models |
| Tensor | Split operations | All-reduce | Wide models |

---

## DDP (Distributed Data Parallel)

Replicates the model on each GPU and synchronizes gradients.

### Configuration

```csharp
using AiDotNet.DistributedTraining;

var config = new DistributedConfig
{
    Backend = DistributedBackend.NCCL,
    WorldSize = 4  // Number of GPUs
};

using var context = DistributedContext.Initialize(config);

// Wrap model with DDP
var ddpModel = DDP.Wrap(model);
```

### Training Loop

```csharp
for (int epoch = 0; epoch < epochs; epoch++)
{
    foreach (var batch in dataLoader)
    {
        var output = ddpModel.Forward(batch.Input);
        var loss = lossFunction.Compute(output, batch.Target);

        ddpModel.Backward(loss);
        optimizer.Step();
        optimizer.ZeroGrad();
    }
}
```

### Multi-Node Setup

```bash
# Node 0 (master)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0
export LOCAL_RANK=0
dotnet run

# Node 1
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4
export LOCAL_RANK=0
dotnet run
```

### Memory Usage

| Model Size | DDP Memory per GPU |
|:-----------|:-------------------|
| 1B | 4 GB |
| 7B | 28 GB |
| 13B | 52 GB (may OOM) |

---

## FSDP (Fully Sharded Data Parallel)

Shards model parameters across GPUs for memory efficiency.

### Configuration

```csharp
using AiDotNet.DistributedTraining.FSDP;

var fsdpConfig = new FSDPConfig<float>
{
    ShardingStrategy = ShardingStrategy.FullShard,
    MixedPrecision = new FSDPMixedPrecisionConfig
    {
        Enabled = true,
        ParameterDtype = DataType.Float32,
        ReduceDtype = DataType.Float32,
        BufferDtype = DataType.BFloat16
    },
    ActivationCheckpointing = new ActivationCheckpointingConfig
    {
        Enabled = true,
        CheckpointInterval = 2
    }
};

var fsdpModel = FSDP<float>.Wrap(model, fsdpConfig);
```

### Sharding Strategies

| Strategy | Description | Memory | Speed |
|:---------|:------------|:-------|:------|
| `NoShard` | No sharding (like DDP) | High | Fast |
| `ShardGradOp` | Shard gradients + optimizer | Medium | Medium |
| `FullShard` | Shard everything | Low | Slower |
| `HybridShard` | Full shard within node | Balanced | Balanced |

### Memory Comparison

| Strategy | 7B Model Memory/GPU (4 GPUs) |
|:---------|:-----------------------------|
| DDP | 28+ GB (OOM) |
| SHARD_GRAD_OP | ~14 GB |
| FULL_SHARD | ~8 GB |
| FULL_SHARD + Checkpointing | ~5 GB |

### Wrapping Policies

```csharp
// Auto-wrap transformer layers
var fsdpConfig = new FSDPConfig<float>
{
    AutoWrapPolicy = new TransformerAutoWrapPolicy
    {
        TransformerLayerClass = typeof(TransformerBlock<float>)
    }
};

// Size-based wrapping
var fsdpConfig = new FSDPConfig<float>
{
    AutoWrapPolicy = new SizeBasedAutoWrapPolicy
    {
        MinNumParams = 100_000_000  // 100M params
    }
};
```

---

## ZeRO Optimization

DeepSpeed-style memory optimization.

### ZeRO Stage 1 (Optimizer State Partitioning)

```csharp
using AiDotNet.DistributedTraining.ZeRO;

var zero1 = new ZeROOptimizer<float>(
    baseOptimizer: new AdamOptimizer<float>(),
    stage: ZeROStage.Stage1);
```

### ZeRO Stage 2 (+ Gradient Partitioning)

```csharp
var zero2 = new ZeROOptimizer<float>(
    baseOptimizer: new AdamOptimizer<float>(),
    stage: ZeROStage.Stage2);
```

### ZeRO Stage 3 (+ Parameter Partitioning)

```csharp
var zero3 = new ZeROOptimizer<float>(
    baseOptimizer: new AdamOptimizer<float>(),
    stage: ZeROStage.Stage3);
```

### Memory Reduction

| Stage | Optimizer States | Gradients | Parameters | Memory Reduction |
|:------|:-----------------|:----------|:-----------|:-----------------|
| Stage 1 | Sharded | Replicated | Replicated | ~4x |
| Stage 2 | Sharded | Sharded | Replicated | ~8x |
| Stage 3 | Sharded | Sharded | Sharded | ~Linear with GPUs |

---

## Pipeline Parallelism

Splits model across GPUs by layers.

### Configuration

```csharp
using AiDotNet.DistributedTraining.Pipeline;

var pipelineConfig = new PipelineConfig
{
    NumStages = 4,
    MicroBatchSize = 4,
    NumMicroBatches = 8
};

var stages = new[]
{
    new PipelineStage(layers: model.Layers[..6], device: 0),
    new PipelineStage(layers: model.Layers[6..12], device: 1),
    new PipelineStage(layers: model.Layers[12..18], device: 2),
    new PipelineStage(layers: model.Layers[18..], device: 3)
};

var pipelineModel = Pipeline.Wrap(model, stages, pipelineConfig);
```

### Scheduling

| Schedule | Description | Bubble Overhead |
|:---------|:------------|:----------------|
| `GPipe` | Simple forward-backward | High |
| `1F1B` | Interleaved forward/backward | Lower |
| `Interleaved1F1B` | Multiple micro-batches | Lowest |

---

## Tensor Parallelism

Splits individual operations across GPUs.

### Configuration

```csharp
using AiDotNet.DistributedTraining.TensorParallel;

var tpConfig = new TensorParallelConfig
{
    WorldSize = 8,
    ParallelMode = ParallelMode.ColumnParallel
};

// Parallel linear layer
var parallelLinear = new ColumnParallelLinear<float>(
    inputDim: 4096,
    outputDim: 16384,
    config: tpConfig);
```

### Parallel Modes

| Mode | Splits | Best For |
|:-----|:-------|:---------|
| `ColumnParallel` | Output features | Linear layers |
| `RowParallel` | Input features | After column parallel |
| `SequenceParallel` | Sequence dimension | Attention |

---

## Using PredictionModelBuilder

```csharp
var result = await new PredictionModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(largeModel)
    .ConfigureOptimizer(new AdamWOptimizer<float>())
    .ConfigureDistributedTraining(new DistributedConfig
    {
        Strategy = DistributedStrategy.FSDP,
        WorldSize = 8,
        ShardingStrategy = ShardingStrategy.FullShard
    })
    .ConfigureGpuAcceleration(new GpuAccelerationConfig
    {
        Enabled = true,
        MixedPrecision = true
    })
    .BuildAsync(trainData, trainLabels);
```

---

## Cloud Training

### vast.ai

```csharp
var config = new DistributedConfig
{
    Backend = DistributedBackend.NCCL,
    WorldSize = int.Parse(Environment.GetEnvironmentVariable("WORLD_SIZE") ?? "1"),
    Rank = int.Parse(Environment.GetEnvironmentVariable("RANK") ?? "0"),
    MasterAddress = Environment.GetEnvironmentVariable("MASTER_ADDR") ?? "localhost",
    MasterPort = int.Parse(Environment.GetEnvironmentVariable("MASTER_PORT") ?? "29500")
};
```

### Azure ML

```yaml
# azure-ml-config.yml
compute:
  instance_type: Standard_NC24ads_A100_v4
  instance_count: 4

distributed:
  type: PyTorch  # Uses NCCL backend
```

### AWS SageMaker

```python
# sagemaker-config.py
estimator = PyTorch(
    entry_point='train.py',
    instance_type='ml.p4d.24xlarge',
    instance_count=4,
    distribution={'smdistributed': {'dataparallel': {'enabled': True}}}
)
```

---

## Checkpointing

### DDP Checkpointing

```csharp
// Save (only on rank 0)
if (context.Rank == 0)
{
    model.SaveCheckpoint("checkpoint.pt");
}
context.Barrier();

// Load (all ranks)
model.LoadCheckpoint("checkpoint.pt");
```

### FSDP Checkpointing

```csharp
// Full state dict (gather to rank 0)
var stateDictConfig = new StateDictConfig
{
    Type = StateDictType.FullStateDict,
    Rank0Only = true
};

fsdpModel.SaveCheckpoint("fsdp_checkpoint.pt", stateDictConfig);

// Sharded state dict (each rank saves own shard)
var shardedConfig = new StateDictConfig
{
    Type = StateDictType.ShardedStateDict
};

fsdpModel.SaveCheckpoint("fsdp_sharded/", shardedConfig);
```

---

## Troubleshooting

### NCCL Timeout

```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_TIMEOUT=300000
```

### Memory Issues

```csharp
// Reduce batch size
// Enable gradient checkpointing
fsdpConfig.ActivationCheckpointing.Enabled = true;

// Use FULL_SHARD
fsdpConfig.ShardingStrategy = ShardingStrategy.FullShard;
```

### Slow Communication

```bash
# Check NVLink topology
nvidia-smi topo -m

# Disable InfiniBand on cloud without IB
export NCCL_IB_DISABLE=1
```

---

## Strategy Selection Guide

| Model Size | GPUs | Recommended Strategy |
|:-----------|:-----|:---------------------|
| < 1B | 1-4 | DDP |
| 1B - 7B | 2-8 | DDP or FSDP ShardGradOp |
| 7B - 13B | 4-8 | FSDP FullShard |
| 13B - 70B | 8-16 | FSDP + ZeRO-3 |
| > 70B | 16+ | FSDP + Pipeline + Tensor |

---

## Best Practices

1. **Start with DDP**: Use FSDP only when necessary
2. **Enable mixed precision**: FP16/BF16 for faster training
3. **Use gradient accumulation**: Increase effective batch size
4. **Enable activation checkpointing**: Trade compute for memory
5. **Profile communication**: Ensure NCCL is performing well
6. **Use appropriate batch size**: Scale with world size
7. **Monitor GPU utilization**: Should be > 90%
