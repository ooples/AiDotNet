using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public class DenseFeatureImportanceTests
{
    [Fact]
    public void DeferredWidth_HasNoAttribution_AndRemainsUnresolved()
    {
        var dense = new DenseLayer<double>(2);
        using var model = new ImportanceProbe(dense);

        Assert.False(dense.IsShapeResolved);
        Assert.Equal(new[] { -1 }, dense.GetInputShape());
        Assert.Empty(model.GetFeatureImportance());
        Assert.False(dense.IsShapeResolved);
        Assert.Equal(new[] { -1 }, dense.GetInputShape());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void MaterializedWeights_AreSummedPerInputFeature_IndependentlyOfBatchAndBias(int rank)
    {
        var dense = new DenseLayer<double>(2);
        using var model = new ImportanceProbe(dense);
        int[] shape = rank == 1 ? new[] { 3 } : rank == 2 ? new[] { 5, 3 } : new[] { 2, 4, 3 };
        _ = dense.Forward(new Tensor<double>(shape));
        // W = [[1,-2], [3,4], [0,-5]], b = [1000,-2000]. Input-feature
        // absolute sums are [3,7,5], normalized by 15. Bias has no input attribution.
        dense.SetParameters(new Vector<double>(new[] { 1.0, -2.0, 3.0, 4.0, 0.0, -5.0, 1000.0, -2000.0 }));

        var importance = model.GetFeatureImportance();

        Assert.Equal(3, importance.Count);
        Assert.Equal(3.0 / 15.0, importance["Feature_0"], 12);
        Assert.Equal(7.0 / 15.0, importance["Feature_1"], 12);
        Assert.Equal(5.0 / 15.0, importance["Feature_2"], 12);
    }

    [Fact]
    public void ZeroWeights_WithNonzeroBias_KeepUniformFeatureAttribution()
    {
        var dense = new DenseLayer<double>(2);
        using var model = new ImportanceProbe(dense);
        _ = dense.Forward(new Tensor<double>(new[] { 2, 3 }));
        dense.SetParameters(new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, -2000.0 }));

        var importance = model.GetFeatureImportance();

        Assert.Equal(3, importance.Count);
        Assert.All(importance.Values, value => Assert.Equal(1.0 / 3.0, value, 12));
    }

    private sealed class ImportanceProbe : NeuralNetworkBase<double>
    {
        public ImportanceProbe(DenseLayer<double> dense) : base(new MeanSquaredErrorLoss<double>())
        {
            Layers.Add(dense);
        }

        protected override void InitializeLayers() { }

        public override ModelMetadata<double> GetModelMetadata() => new();
    }
}
