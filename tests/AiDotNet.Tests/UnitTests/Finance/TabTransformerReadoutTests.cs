using AiDotNet.Enums;
using AiDotNet.Finance.Risk;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// TabTransformer (Huang et al. 2020) reduces the contextual embeddings to one vector per sample before
/// its MLP, so it predicts one row per sample and trains against [batch, outputs] targets.
/// </summary>
public class TabTransformerReadoutTests
{
    private static Tensor<double> Random(int seed, params int[] shape)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++)
            t.SetFlat(i, random.NextDouble() - 0.5);
        return t;
    }

    private static TabTransformer<double> Create() =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
                inputSize: 6, outputSize: 1),
            new TabTransformerOptions<double> { NumFeatures = 6, HiddenDimension = 16, NumHeads = 4, EmbeddingDimension = 16 });

    [Fact]
    public void Predict_ReturnsOneRowPerSample()
    {
        var prediction = Create().Predict(Random(1, 5, 6));

        Assert.Equal(new[] { 5, 1 }, prediction.Shape.ToArray());
    }

    [Fact]
    public void Training_OnPerSampleTargets_LowersTheLoss()
    {
        var model = Create();
        var x = Random(2, 16, 6);
        var y = new Tensor<double>(new[] { 16, 1 });
        for (int r = 0; r < 16; r++)
            y[r, 0] = x[r, 0] - 0.5 * x[r, 3];

        double Loss()
        {
            var p = model.Predict(x);
            double sum = 0;
            for (int r = 0; r < 16; r++)
                sum += (p[r, 0] - y[r, 0]) * (p[r, 0] - y[r, 0]);
            return sum / 16;
        }

        double before = Loss();
        for (int step = 0; step < 60; step++)
            model.Train(x, y);
        double after = Loss();

        Assert.True(after < 0.8 * before, $"loss went from {before} to {after}");
    }
}
