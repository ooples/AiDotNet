// AiDotNet — Fully Sharded Data Parallel (FSDP)
//
// Like DDP, but the model's parameters, gradients, and optimizer state are
// SHARDED across ranks to fit models too large for one device. It is selected
// through the same AiModelBuilder facade by passing DistributedStrategy.FSDP to
// ConfigureDistributedTraining; BuildAsync wraps the model accordingly. This
// sample uses an in-memory world-size-1 backend so it runs anywhere.

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;     // InMemoryCommunicationBackend
using AiDotNet.Enums;                    // DistributedStrategy
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Fully Sharded Data Parallel (FSDP) Training ===\n");

// ── Communication backend (single process: world size 1) ───────────────────
var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
Console.WriteLine($"Backend: in-memory  |  world size: {backend.WorldSize}  |  rank: {backend.Rank}");
Console.WriteLine("Strategy: FSDP (shard parameters, gradients, optimizer state)\n");

// ── Model + synthetic data ─────────────────────────────────────────────────
const int features = 16;
const int numClasses = 3;
const int n = 48;

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: features, numClasses: numClasses, complexity: NetworkComplexity.Simple));

var rng = new Random(42);
var trainX = new Tensor<double>(new[] { n, features });
var trainY = new Tensor<double>(new[] { n, numClasses });
for (int i = 0; i < n; i++)
{
    for (int j = 0; j < features; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % numClasses }] = 1.0;
}

// ── Train fully-sharded through the facade ─────────────────────────────────
Console.WriteLine("Training through AiModelBuilder.ConfigureDistributedTraining (FSDP) ...");
try
{
    var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
        .ConfigureModel(model)
        .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
        .ConfigureDistributedTraining(backend, DistributedStrategy.FSDP)
        .BuildAsync();

    Console.WriteLine("  Training complete.");
    var prediction = result.Predict(trainX);
    Console.WriteLine($"  Prediction shape: [{string.Join(", ", prediction.Shape)}]");
}
catch (Exception ex)
{
    Console.WriteLine($"  Distributed training reported: {ex.Message}");
}

Console.WriteLine(@"
FSDP shards each layer's parameters across ranks and gathers them just-in-time
for the forward/backward pass, so the per-rank memory footprint scales down with
the number of ranks — enabling models far larger than a single device can hold.
");

Console.WriteLine("=== Sample Complete ===");
