# Distributed Data Parallel (DDP) Training

This sample demonstrates how to train models across multiple GPUs using Distributed Data Parallel.

## What You'll Learn

- How to configure distributed training with `AiModelBuilder`
- How to use NCCL backend for GPU communication
- How to scale training across multiple GPUs
- How to monitor distributed training progress

## What is DDP?

Distributed Data Parallel (DDP) replicates the model on each GPU and splits the batch across them:

```
┌─────────────────────────────────────────────────────────┐
│                    Training Batch                        │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │ Mini-1  │  │ Mini-2  │  │ Mini-3  │  │ Mini-4  │   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │
│       │            │            │            │         │
│       ▼            ▼            ▼            ▼         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │   │
│  │ Model   │  │ Model   │  │ Model   │  │ Model   │   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │
│       │            │            │            │         │
│       └────────────┴─────┬──────┴────────────┘         │
│                          │                              │
│                    AllReduce                            │
│                   (sync grads)                          │
│                          │                              │
│                    ┌─────┴─────┐                        │
│                    │  Update   │                        │
│                    │  Weights  │                        │
│                    └───────────┘                        │
└─────────────────────────────────────────────────────────┘
```

## Benefits of DDP

- **Near-linear scaling**: 4 GPUs ≈ 4x training speed
- **Synchronized gradients**: All GPUs have identical weights
- **Memory efficient**: Only stores one model copy + gradients

## Running the Sample

### Prerequisites

- Multiple NVIDIA GPUs with CUDA
- NCCL library installed

### Single Node, Multi-GPU

```bash
dotnet run -- --gpus 0,1,2,3
```

### Multi-Node

```bash
# Node 0 (master)
dotnet run -- --gpus 0,1 --world-size 4 --rank 0 --master-addr 192.168.1.100

# Node 1
dotnet run -- --gpus 0,1 --world-size 4 --rank 2 --master-addr 192.168.1.100
```

## Expected Output

```
=== AiDotNet Distributed Training (DDP) ===

Initializing distributed environment...
  Backend: NCCL
  World size: 4
  Local rank: 0
  Global rank: 0

Loading model on GPU 0...
  Model: ResNet50
  Parameters: 25.6M

Wrapping model with DDP...
  Bucket size: 25MB
  Gradient compression: Enabled

Training...
  Epoch 1/10
    Batch 100/500 | Loss: 2.341 | Throughput: 3,240 img/s
    Batch 200/500 | Loss: 1.892 | Throughput: 3,312 img/s
    ...
    Epoch 1 complete | Loss: 1.234 | Time: 45.2s

  Epoch 10/10
    Epoch 10 complete | Loss: 0.156 | Time: 42.8s

Training complete!
  Total time: 7m 32s
  Average throughput: 3,285 img/s
  Scaling efficiency: 94.2%
```

## Code Highlights

```csharp
var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(resnet50)
    .ConfigureOptimizer(new AdamWOptimizer<float>(learningRate: 0.001f))
    .ConfigureDistributedTraining(
        strategy: DistributedStrategy.DDP,
        backend: new NCCLCommunicationBackend(),
        configuration: new DDPConfiguration
        {
            BucketSizeMB = 25,
            GradientCompression = true,
            FindUnusedParameters = false
        })
    .ConfigureGpuAcceleration(new GpuAccelerationConfig
    {
        DeviceIds = new[] { 0, 1, 2, 3 }
    })
    .ConfigureMixedPrecision()  // FP16 for faster training
    .BuildAsync(trainData);
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `BucketSizeMB` | 25 | Gradient bucket size for AllReduce |
| `GradientCompression` | false | Enable gradient compression |
| `FindUnusedParameters` | false | Find and ignore unused params |
| `BroadcastBuffers` | true | Sync batch norm buffers |

## Scaling Guidelines

| GPUs | Batch Size | Learning Rate | Expected Speedup |
|------|------------|---------------|------------------|
| 1 | 32 | 0.001 | 1x |
| 2 | 64 | 0.002 | ~1.9x |
| 4 | 128 | 0.004 | ~3.7x |
| 8 | 256 | 0.008 | ~7.2x |

## Troubleshooting

**NCCL timeout**: Increase `NCCL_TIMEOUT` environment variable
**OOM errors**: Reduce batch size per GPU
**Slow communication**: Check InfiniBand/NVLink connectivity

## Next Steps

- [FSDP](../FSDP/) - Train larger models with memory sharding
- [PipelineParallel](../PipelineParallel/) - Split model across GPUs
