// AiDotNet — Distributed Data Parallel (DDP)
//
// Data-parallel training through the AiModelBuilder facade. A communication
// backend + DistributedStrategy.DDP are supplied via ConfigureDistributedTraining;
// BuildAsync wraps the model for synchronized-gradient training. This sample uses
// an in-memory backend with world size 1 (so it runs anywhere), but the identical
// facade call scales to multiple ranks with a real backend (NCCL/Gloo/MPI).

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.DistributedTraining;     // InMemoryCommunicationBackend
using AiDotNet.Enums;                    // DistributedStrategy
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Distributed Data Parallel (DDP) Training ===\n");

// ── Communication backend (single process: world size 1) ───────────────────
var backend = new InMemoryCommunicationBackend<double>(rank: 0, worldSize: 1);
Console.WriteLine($"Backend: in-memory  |  world size: {backend.WorldSize}  |  rank: {backend.Rank}\n");

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

// ── Train data-parallel through the facade ─────────────────────────────────
Console.WriteLine("Training through AiModelBuilder.ConfigureDistributedTraining (DDP) ...");
try
{
    var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
        .ConfigureModel(model)
        .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
        .ConfigureDistributedTraining(backend, DistributedStrategy.DDP)
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
DDP replicates the model on every rank and averages (all-reduces) gradients each
step so all replicas stay in sync. Scale out by giving ConfigureDistributedTraining
a multi-rank backend (NCCL on GPUs, Gloo/MPI on CPUs) — the facade call is the same.
");

Console.WriteLine("=== Sample Complete ===");
