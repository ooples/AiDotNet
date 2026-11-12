using System;
using System.Linq;
using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Mean-ablation feature importance, used by SAINT and TabTransformer.
/// </summary>
public class TabularFeatureImportanceTests
{
    [Fact]
    public void MeanAblation_MatchesHandDerivedLinearModel()
    {
        // y = 2 x0 + 0 x1 - x2 over rows [1, 5, 3] and [3, 5, 1]; column means [2, 5, 2].
        // Ablating x0 moves each row by |2| * 1, total 4; x1 moves nothing; x2 moves each row by 1, total 2.
        // Normalized: [4/6, 0, 2/6].
        var weights = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 2.0, 0.0, -1.0 }));
        Tensor<double> Predict(Tensor<double> x) => AiDotNetEngine.Current.TensorMatMul(x, weights);
        var batch = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 1.0, 5.0, 3.0, 3.0, 5.0, 1.0 }));

        var importance = TabularMetrics.MeanAblationImportance(Predict, batch);

        Assert.Equal(3, importance.Length);
        Assert.Equal(4.0 / 6.0, importance[0], 12);
        Assert.Equal(0.0, importance[1], 12);
        Assert.Equal(2.0 / 6.0, importance[2], 12);
    }

    [Fact]
    public void MeanAblation_IsUniformWhenNothingChanges()
    {
        var weights = new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new[] { 1.0, 1.0 }));
        Tensor<double> Predict(Tensor<double> x) => AiDotNetEngine.Current.TensorMatMul(x, weights);
        var constantColumns = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 1.0, 2.0, 1.0, 2.0 }));

        var importance = TabularMetrics.MeanAblationImportance(Predict, constantColumns);

        Assert.Equal(new[] { 0.5, 0.5 }, importance);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Network_AttributesTheLastPredictedBatch(bool saint)
    {
        TabularNeuralNetworkBase<double> network = saint
            ? new SAINTNetwork<double>()
            : new TabTransformerNetwork<double>();
        int features = saint ? 10 : 16;

        // Before any prediction there is nothing to attribute: uniform.
        var before = network.GetFeatureImportance();
        Assert.Equal(features, before.Count);
        Assert.All(before.Values, value => Assert.Equal(1.0 / features, value, 12));

        var batch = new Tensor<double>(new[] { 4, features });
        for (int i = 0; i < batch.Length; i++) batch[i] = Math.Sin(0.7 * i + 0.3) * (1 + i % 3);
        _ = network.Predict(batch);

        var after = network.GetFeatureImportance();
        Assert.Equal(features, after.Count);
        Assert.Equal(1.0, after.Values.Sum(), 10);
        Assert.All(after.Values, value => Assert.True(value >= 0.0));
    }
}
