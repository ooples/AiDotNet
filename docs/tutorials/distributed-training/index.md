---
layout: default
title: Distributed Training
parent: Tutorials
nav_order: 9
has_children: true
permalink: /tutorials/distributed-training/
---

# Distributed Training Tutorial
{: .no_toc }

Scale your training across multiple GPUs and nodes.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AiDotNet provides 10+ distributed training strategies:
- **DDP**: Distributed Data Parallel - replicate model, split data
- **FSDP**: Fully Sharded Data Parallel - shard model across GPUs
- **ZeRO**: Zero Redundancy Optimizer (1/2/3) - memory optimization
- **Pipeline**: Pipeline Parallelism - split model by layers
- **Tensor**: Tensor Parallelism - split individual operations

---

## DDP (Distributed Data Parallel)

Best for: Models that fit on a single GPU

```csharp
using AiDotNet.DistributedTraining;

// Initialize process group
var config = new DistributedConfig
{
    Backend = DistributedBackend.NCCL,
    WorldSize = 4  // 4 GPUs
};

using var context = DistributedContext.Initialize(config);

// Wrap model
var ddpModel = DDP.Wrap(model);

// Training loop (same as single GPU)
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
dotnet run

# Node 1
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4
dotnet run
```

---

## FSDP (Fully Sharded Data Parallel)

Best for: Large models that don't fit on a single GPU

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

// Wrap model with FSDP
var fsdpModel = FSDP<float>.Wrap(model, fsdpConfig);

// Memory is now sharded across GPUs!
```

### Memory Comparison

| Strategy | 7B Model Memory/GPU |
|:---------|:--------------------|
| Single GPU | 28+ GB (OOM) |
| DDP (4 GPU) | 28+ GB (OOM) |
| FSDP SHARD_GRAD_OP | ~14 GB |
| FSDP FULL_SHARD | ~8 GB |
| FSDP + Checkpointing | ~5 GB |

---

## ZeRO Optimization

DeepSpeed-style memory optimization:

```csharp
using AiDotNet.DistributedTraining.ZeRO;

// ZeRO Stage 1: Partition optimizer states
var zero1 = new ZeROOptimizer<float>(
    baseOptimizer: new AdamOptimizer<float>(),
    stage: ZeROStage.Stage1);

// ZeRO Stage 2: + Partition gradients
var zero2 = new ZeROOptimizer<float>(
    baseOptimizer: new AdamOptimizer<float>(),
    stage: ZeROStage.Stage2);

// ZeRO Stage 3: + Partition parameters
var zero3 = new ZeROOptimizer<float>(
    baseOptimizer: new AdamOptimizer<float>(),
    stage: ZeROStage.Stage3);
```

---

## Pipeline Parallelism

Split model across GPUs by layers:

```csharp
using AiDotNet.DistributedTraining.Pipeline;

var pipelineConfig = new PipelineConfig
{
    NumStages = 4,
    MicroBatchSize = 4,
    NumMicroBatches = 8
};

// Define pipeline stages (which layers go where)
var stages = new[]
{
    new PipelineStage(layers: model.Layers[..6], device: 0),
    new PipelineStage(layers: model.Layers[6..12], device: 1),
    new PipelineStage(layers: model.Layers[12..18], device: 2),
    new PipelineStage(layers: model.Layers[18..], device: 3)
};

var pipelineModel = Pipeline.Wrap(model, stages, pipelineConfig);
```

---

## Using AiModelBuilder

```csharp
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
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

### vast.ai Setup

```csharp
// Configure for vast.ai multi-GPU instances
var config = new DistributedConfig
{
    Backend = DistributedBackend.NCCL,
    WorldSize = Environment.GetEnvironmentVariable("WORLD_SIZE"),
    Rank = Environment.GetEnvironmentVariable("RANK"),
    MasterAddress = Environment.GetEnvironmentVariable("MASTER_ADDR"),
    MasterPort = Environment.GetEnvironmentVariable("MASTER_PORT")
};
```

### Azure ML

```yaml
# azure-ml-config.yml
compute:
  instance_type: Standard_NC24ads_A100_v4
  instance_count: 4

distributed:
  type: PyTorch  # Uses NCCL
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
// FSDP handles distributed state automatically
fsdpModel.SaveCheckpoint("fsdp_checkpoint.pt");

// Load
fsdpModel.LoadCheckpoint("fsdp_checkpoint.pt");
```

---

## Best Practices

1. **Start with DDP**: Use FSDP only if model doesn't fit
2. **Use gradient accumulation**: Increase effective batch size
3. **Enable mixed precision**: Faster training, less memory
4. **Monitor GPU utilization**: Use nvidia-smi or dcgm
5. **Profile communication**: Ensure NCCL is performing well

### Gradient Accumulation

```csharp
int accumulationSteps = 4;
int effectiveBatchSize = batchSize * accumulationSteps * worldSize;

for (int step = 0; step < accumulationSteps; step++)
{
    var loss = ComputeLoss(batch) / accumulationSteps;
    loss.Backward();
}

optimizer.Step();
optimizer.ZeroGrad();
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
// Use FSDP FULL_SHARD
```

### Slow Communication

```bash
# Check if using NVLink
nvidia-smi topo -m

# Use NCCL_IB_DISABLE for cloud without InfiniBand
export NCCL_IB_DISABLE=1
```

---

## Next Steps

- [DDP Sample](/samples/advanced/DistributedTraining/DDP/)
- [FSDP Sample](/samples/advanced/DistributedTraining/FSDP/)
- [Distributed Training API Reference](/api/AiDotNet.DistributedTraining/)
