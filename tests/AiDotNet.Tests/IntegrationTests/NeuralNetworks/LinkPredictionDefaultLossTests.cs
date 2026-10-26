using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Tasks.Graph;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// A link prediction model outputs raw scores, and its default loss used to be probability-space binary
/// cross-entropy, which clamps its input to [eps, 1 - eps]. Whenever every output fell outside (0, 1) the gradient
/// was exactly zero and Train never moved a weight: 16 of 200 unseeded initialisations, which is what failed
/// GraphTaskFusedTrainingTests on CI by chance. The default now takes logits.
/// </summary>
public class LinkPredictionDefaultLossTests
{
    [Fact(Timeout = 120000)]
    public async Task TrainingMovesTheWeights_FromEveryInitialisation()
    {
        await Task.Yield();
        var stuck = new List<int>();
        for (int seed = 1; seed <= 40; seed++)
        {
            var model = new LinkPredictionModel<double>(
                new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
                    taskType: NeuralNetworkTaskType.BinaryClassification, complexity: NetworkComplexity.Simple,
                    inputSize: 3, outputSize: 1) { RandomSeed = seed },
                options: new LinkPredictionOptions { HiddenDim = 4, EmbeddingDim = 3, NumLayers = 2, DropoutRate = 0.0 });
            var adjacency = new Tensor<double>(new[] { 5, 5 });
            for (int i = 0; i < 5; i++)
            {
                adjacency[i, i] = 1.0;
                adjacency[i, (i + 1) % 5] = 1.0;
                adjacency[(i + 1) % 5, i] = 1.0;
            }

            model.SetAdjacencyMatrix(adjacency);
            var features = new Tensor<double>(new[] { 5, 3 });
            for (int i = 0; i < features.Length; i++) features[i] = Math.Sin(0.41 * i);
            var shape = model.Predict(features)._shape;
            var before = model.GetParameters().ToArray();
            for (int step = 0; step < 4; step++)
            {
                var target = new Tensor<double>(shape);
                target[step % target.Length] = 1.0;
                model.Train(features, target);
            }

            var after = model.GetParameters().ToArray();
            double moved = 0.0;
            for (int i = 0; i < after.Length; i++) moved = Math.Max(moved, Math.Abs(after[i] - before[i]));
            if (moved == 0.0) stuck.Add(seed);
        }

        Assert.True(stuck.Count == 0, $"Training left every weight unchanged for seeds {string.Join(", ", stuck)}.");
    }
}
