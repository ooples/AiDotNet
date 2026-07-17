using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActiveLearning;
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
/// Covers the diversity-aware active-learning selection: the BADGE-style batch beats naive top-N
/// uncertainty by avoiding redundant picks, and ConfigureActiveLearning surfaces the selection on Build.
/// </summary>
public class ActiveLearningSelectionTests
{
    [Fact]
    public void BatchActiveLearner_AvoidsRedundantBatch_ThatNaiveTopNWouldPick()
    {
        // Samples 0,1,2 are near-identical (cluster A) and the most uncertain; 3,4 point a different way
        // (cluster B); 5 is elsewhere. Naive top-2 uncertainty would pick {0,1} — two near-duplicates.
        var informativeness = new Vector<double>(new[] { 0.90, 0.88, 0.86, 0.70, 0.50, 0.10 });
        var rep = new Matrix<double>(6, 2);
        double[][] rows =
        {
            new[] { 1.0, 0.0 }, new[] { 0.99, 0.01 }, new[] { 0.98, 0.02 }, // cluster A
            new[] { 0.0, 1.0 }, new[] { 0.01, 0.99 },                        // cluster B
            new[] { 0.7, 0.7 },                                              // elsewhere
        };
        for (int i = 0; i < 6; i++) { rep[i, 0] = rows[i][0]; rep[i, 1] = rows[i][1]; }

        var selection = new BatchActiveLearner<double>(diversityWeight: 1.0)
            .Select(informativeness, rep, batchSize: 2, "test", "InputFeatures");

        Assert.Equal(2, selection.SelectedIndices.Length);
        // First pick is the most uncertain (cluster A).
        Assert.Equal(0, selection.SelectedIndices[0]);
        // Second pick must come from a DIFFERENT region, not the redundant cluster-A neighbours 1/2.
        int second = selection.SelectedIndices[1];
        Assert.True(second is 3 or 4 or 5, $"diversity failed: picked redundant sample {second}");
        Assert.DoesNotContain(1, selection.SelectedIndices);
        Assert.DoesNotContain(2, selection.SelectedIndices);
    }

    [Fact]
    public void BatchActiveLearner_ZeroDiversity_ReducesToTopNUncertainty()
    {
        var informativeness = new Vector<double>(new[] { 0.90, 0.88, 0.86, 0.70, 0.50 });
        var rep = new Matrix<double>(5, 2);
        for (int i = 0; i < 5; i++) { rep[i, 0] = 1.0; rep[i, 1] = 0.0; }

        var selection = new BatchActiveLearner<double>(diversityWeight: 0.0)
            .Select(informativeness, rep, batchSize: 3, "test", "InputFeatures");

        // With no diversity penalty, it is exactly the top-3 most uncertain.
        Assert.Equal(new[] { 0, 1, 2 }, selection.SelectedIndices);
    }

    private sealed class DecreasingInformativenessStrategy : IActiveLearningStrategy<double>
    {
        public string Name => "decreasing-test";
        public bool UseBatchDiversity { get; set; }
        public int[] SelectSamples(IFullModel<double, Tensor<double>, Tensor<double>> model, Tensor<double> pool, int batchSize)
            => Enumerable.Range(0, batchSize).ToArray();
        public Vector<double> ComputeInformativenessScores(IFullModel<double, Tensor<double>, Tensor<double>> model, Tensor<double> pool)
        {
            int n = pool.Shape[0];
            var v = new Vector<double>(n);
            for (int i = 0; i < n; i++) v[i] = n - i;
            return v;
        }
        public System.Collections.Generic.Dictionary<string, double> GetSelectionStatistics() => new();
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureActiveLearning_SurfacesSelectionOnBuild()
    {
        const int cols = 4, outs = 2, rows = 40;
        var x = new Tensor<double>(new[] { rows, cols });
        var y = new Tensor<double>(new[] { rows, outs });
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2);
            for (int o = 0; o < outs; o++) y[i, o] = Math.Cos((i + o) * 0.2);
        }

        var pool = new Tensor<double>(new[] { 12, cols });
        for (int i = 0; i < 12; i++)
            for (int j = 0; j < cols; j++) pool[i, j] = Math.Sin((i * 2 + j) * 0.3);

        var layers = new System.Collections.Generic.List<ILayer<double>>
        {
            new DenseLayer<double>(6, (IActivationFunction<double>)new ReLUActivation<double>()),
            new DenseLayer<double>(outs),
        };
        var arch = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: cols, outputSize: outs, layers: layers);
        var model = new NeuralNetwork<double>(arch);

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureActiveLearning(new DecreasingInformativenessStrategy(), unlabeledPool: pool, batchSize: 3)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .BuildAsync();

        Assert.NotNull(result.ActiveLearningSelection);
        Assert.Equal(3, result.ActiveLearningSelection?.SelectedIndices.Length ?? -1);
        Assert.Equal(12, result.ActiveLearningSelection.Ranking.Count);
        Assert.Equal("decreasing-test", result.ActiveLearningSelection.StrategyName);
        // The three-tier cascade must have chosen one of the recognized representation spaces.
        Assert.Contains(result.ActiveLearningSelection.RepresentationSpace,
            new[] { "GradientEmbedding", "ModelRepresentation", "InputFeatures" });
    }
}
