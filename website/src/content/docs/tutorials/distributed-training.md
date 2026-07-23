---
title: "Distributed Training"
description: "Multi-GPU and multi-node training."
order: 10
section: "Tutorials"
---

Scale training across workers with `ConfigureDistributedTraining(backend, strategy)`. You provide a communication backend and a `DistributedStrategy`; the facade synchronizes gradients across the group each step.

## Strategies

- **`DistributedStrategy.DDP`** — Distributed Data Parallel. Replicate the model, all-reduce gradients. Best when the model fits on one device.
- **`DistributedStrategy.FSDP`** — Fully Sharded Data Parallel. Shard parameters across workers when the model is too large to replicate.
- **`DistributedStrategy.PipelineParallel`** — split a very deep model into stages across devices.

## Data Parallel (DDP)

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// One backend per worker. worldSize 1 here for a runnable single-process example.
var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 32, numClasses: 4, complexity: NetworkComplexity.Simple));

var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 64, 32 });
var trainY = new Tensor<double>(new[] { 64, 4 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 32; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureDistributedTraining(backend, DistributedStrategy.DDP)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"DDP training complete; output [{string.Join(", ", result.Predict(trainX).Shape)}]");
```

## Fully Sharded (FSDP)

When the model is too large to replicate, switch the strategy — nothing else changes.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 64, numClasses: 8, complexity: NetworkComplexity.Medium));

var rng = new Random(7);
var trainX = new Tensor<double>(new[] { 64, 64 });
var trainY = new Tensor<double>(new[] { 64, 8 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 64; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 8 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureDistributedTraining(backend, DistributedStrategy.FSDP)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("FSDP training complete.");
```

## Launching Multiple Workers

The examples above run a single process (`worldSize: 1`). In a real run, launch one process per worker; each constructs a backend with its own `rank` from the environment and the shared `worldSize`.

```csharp
using AiDotNet.DistributedTraining;

int rank = int.Parse(Environment.GetEnvironmentVariable("RANK") ?? "0");
int worldSize = int.Parse(Environment.GetEnvironmentVariable("WORLD_SIZE") ?? "1");

var backend = new InMemoryCommunicationBackend<double>(rank: rank, worldSize: worldSize);
Console.WriteLine($"Worker {rank} of {worldSize}");
```

## Best Practices

1. **Start with DDP**: simplest and covers most workloads.
2. **Use FSDP when memory-bound**: shard parameters only when the model won't fit replicated.
3. **Scale the learning rate**: larger effective batch sizes usually want a higher LR.
4. **Keep workers identical**: same model, same seeds, consistent data sharding.

## Notes

The facade exposes distributed training via `ConfigureDistributedTraining(backend, strategy)`. Lower-level building blocks — tensor-parallel layers, explicit activation-checkpointing configuration, and manual checkpoint/barrier APIs — are not part of the facade surface today.

## Next Steps

- [Distributed Strategies Reference](/docs/reference/distributed-strategies/)
- [LLM Fine-tuning Tutorial](/docs/tutorials/llm-fine-tuning/)
