using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that a generic query strategy configured via ConfigureQueryStrategy is run over its
/// per-sample pool and surfaces a diversity-aware selection on the built result.
/// </summary>
public class ConfiguredQueryStrategyTests
{
    private sealed class DecreasingScoreStrategy : IQueryStrategy<double, Tensor<double>, Tensor<double>>
    {
        public string Name => "decreasing-query";
        public string Description => "test strategy with decreasing scores";
        public Vector<double> ComputeScores(
            IFullModel<double, Tensor<double>, Tensor<double>> model,
            IDataset<double, Tensor<double>, Tensor<double>> unlabeledPool)
        {
            int n = unlabeledPool.Count;
            var v = new Vector<double>(n);
            for (int i = 0; i < n; i++) v[i] = n - i;
            return v;
        }
        public int[] SelectSamples(
            IFullModel<double, Tensor<double>, Tensor<double>> model,
            IDataset<double, Tensor<double>, Tensor<double>> unlabeledPool, int batchSize)
            => Enumerable.Range(0, batchSize).ToArray();
        public void UpdateState(int[] newlyLabeledIndices, Tensor<double>[] labels) { }
        public void Reset() { }
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureQueryStrategy_SurfacesSelectionOnBuild()
    {
        const int cols = 4, outs = 2, rows = 40;
        var x = new Tensor<double>(new[] { rows, cols });
        var y = new Tensor<double>(new[] { rows, outs });
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2);
            for (int o = 0; o < outs; o++) y[i, o] = Math.Cos((i + o) * 0.2);
        }

        // Per-sample pool: each entry is one sample as a [1, cols] tensor.
        var pool = new List<Tensor<double>>();
        for (int i = 0; i < 10; i++)
        {
            var sample = new Tensor<double>(new[] { 1, cols });
            for (int j = 0; j < cols; j++) sample[0, j] = Math.Sin((i * 2 + j) * 0.3);
            pool.Add(sample);
        }

        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(6, (IActivationFunction<double>)new ReLUActivation<double>()),
            new DenseLayer<double>(outs),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: cols, outputSize: outs, layers: layers);
        var model = new NeuralNetwork<double>(arch);

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureQueryStrategy(new DecreasingScoreStrategy(), unlabeledPool: pool, batchSize: 3)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result.ActiveLearningSelection);
        Assert.Equal(3, result.ActiveLearningSelection?.SelectedIndices.Length ?? -1);
        Assert.Equal(10, result.ActiveLearningSelection.Ranking.Count);
        Assert.Equal("decreasing-query", result.ActiveLearningSelection.StrategyName);
    }
}
