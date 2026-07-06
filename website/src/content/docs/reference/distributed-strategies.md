---
title: "Distributed Strategies"
description: "Distributed training strategy reference."
order: 6
section: "Reference"
---


Train across multiple workers through the facade with `ConfigureDistributedTraining(backend, strategy)`. You supply a communication backend and a `DistributedStrategy`; AiDotNet handles gradient synchronization each step so replicas stay in sync.

---

## Strategies

| Strategy | Description | Use When |
|:---------|:------------|:---------|
| `DistributedStrategy.DDP` | Distributed Data Parallel — replicate the model, all-reduce gradients | The model fits on one device (covers ~90% of cases) |
| `DistributedStrategy.FSDP` | Fully Sharded Data Parallel — shard parameters across workers | The model is too large for one device |
| `DistributedStrategy.ZeRO3` | ZeRO Stage 3 — full sharding (FSDP terminology) | You prefer ZeRO terminology |
| `DistributedStrategy.PipelineParallel` | Split the model into stages across devices | Very deep models |

---

## Data Parallel (DDP)

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// One backend per process. rank/worldSize identify this worker in the group;
// here a single-process group (worldSize 1) for a runnable example.
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

For models too large to replicate, switch the strategy to `FSDP` — everything else is identical.

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

---

## Scaling Out

The single-process examples above use `worldSize: 1`. In a real multi-worker run, each process constructs a backend with its own `rank` and the shared `worldSize`, launched once per worker (e.g. via your scheduler or `torchrun`-style launcher). `ConfigureDistributedTraining` all-reduces gradients across the group every step.

```csharp
using AiDotNet.DistributedTraining;

// In each worker process, rank comes from the launch environment.
int rank = int.Parse(Environment.GetEnvironmentVariable("RANK") ?? "0");
int worldSize = int.Parse(Environment.GetEnvironmentVariable("WORLD_SIZE") ?? "1");

var backend = new InMemoryCommunicationBackend<double>(rank: rank, worldSize: worldSize);
Console.WriteLine($"Worker {rank} of {worldSize}");
```

---

## Best Practices

1. **Start with DDP**: it covers most workloads and is the simplest to reason about.
2. **Reach for FSDP when memory-bound**: shard parameters only when the model won't fit replicated.
3. **Scale the learning rate**: larger effective batch sizes (more workers) usually want a higher LR.
4. **Keep workers identical**: same model, same seed handling, same data sharding.

---

## Notes

The facade exposes distributed training through `ConfigureDistributedTraining(backend, strategy)` over a communication backend. Lower-level building blocks shown in some references — tensor-parallel layers (column/row-parallel linear), explicit activation-checkpointing configuration, and manual checkpoint/barrier APIs — are not part of the facade surface today.
