using System;
using System.Linq;
using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class Elementwise02RegressionTests
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;

    [Fact]
    public void RbfSpread_LargeHighDimensionalCloud_MatchesClosedForm()
    {
        const int count = 512, dimensions = 768;
        var layer = new RBFLayer<double>(1, 2);
        var centers = new Tensor<double>(new[] { count, dimensions });
        for (int i = (count - 1) * dimensions; i < centers.Length; i++) centers.SetFlat(i, 1.0);
        var method = layer.GetType().GetMethod("CenterSpreadWidth", Private, null,
            new[] { typeof(Tensor<double>), typeof(int) }, null)!;
        double width = (double)method.Invoke(layer, new object[] { centers, count })!;
        // The most distant pair differs by 1 in each dimension: d_max^2 = dimensions.
        Assert.Equal(Math.Sqrt(dimensions / (2.0 * count)), width, 12);
    }

    [Fact]
    public void RbfSpread_HalfLargeCoordinates_UsesWideSquaredDistances()
    {
        var layer = new RBFLayer<Half>(1, 2);
        var centers = new Tensor<Half>(new[] { (Half)(-1000), (Half)1000 }, new[] { 2, 1 });
        var method = layer.GetType().GetMethod("CenterSpreadWidth", Private, null,
            new[] { typeof(Tensor<Half>), typeof(int) }, null)!;
        Assert.Equal((Half)1000, (Half)method.Invoke(layer, new object[] { centers, 2 })!);
    }

    [Fact]
    public void MeshSort_LargeEdgeSet_UsesAscendingStableOrder()
    {
        const int count = 16384;
        var layer = new MeshPoolLayer<double>(1, 1);
        var values = new Tensor<double>(Enumerable.Range(0, count).Select(i => (double)(count - i)).ToArray(), new[] { count });
        var method = layer.GetType().GetMethod("SortEdgesByImportance", Private)!;
        var order = (int[])method.Invoke(layer, new object[] { values, count })!;
        Assert.Equal(Enumerable.Range(0, count).Reverse(), order);
        var tied = new Tensor<double>(new[] { 2.0, double.NaN, 1.0, 1.0, double.NaN, 2.0 }, new[] { 6 });
        Assert.Equal(new[] { 1, 4, 2, 3, 0, 5 }, (int[])method.Invoke(layer, new object[] { tied, 6 })!);
    }

    [Fact]
    public void HalfCausalBias_DistinguishesAdjacentKeysBeyond2048()
    {
        var layer = new MultiHeadAttentionLayer<Half>(1, 2) { UseCausalMask = true };
        var method = layer.GetType().GetMethod("AttentionDropoutLogitBias", Private)!;
        var bias = (Tensor<Half>)method.Invoke(layer, new object[] { 1, 2, 2050 })!;
        Assert.Equal((Half)0, bias[0, 0, 2048]);
        Assert.True(Half.IsNegativeInfinity(bias[0, 0, 2049]));
        Assert.Equal((Half)0, bias[0, 1, 2049]);
    }

    [Fact]
    public void HalfLayoutCoordinates_HandleNanAndRoundTiesToEven()
    {
        var layer = new LayoutEmbeddingLayer<Half>(vocabSize: 4, hiddenDim: 2,
            maxSequenceLength: 4, maxPosition2D: 8, includeTokens: false);
        var method = layer.GetType().GetMethod("ClampCoordinate", Private)!;
        var input = new Tensor<Half>(new[] { Half.NaN, (Half)2.5, (Half)3.5, (Half)(-1), (Half)20 }, new[] { 5 });
        var result = (Tensor<Half>)method.Invoke(layer, new object[] { input })!;
        Assert.Equal(new[] { (Half)0, (Half)2, (Half)4, (Half)0, (Half)7 }, result.ToArray());
    }

    [Theory]
    [InlineData(1e200)]
    [InlineData(1e-320)]
    public void ReservoirRadius_ExtremeScales_PreserveComplexModulus(double scale)
    {
        var layer = new ReservoirLayer<double>(1, 2, connectionProbability: 0);
        var matrix = new Tensor<double>(new[] { 0.0, -scale, scale, 0.0 }, new[] { 2, 2 });
        Assert.InRange(layer.ComputeMaxEigenvalue(matrix) / scale, 0.9999999999, 1.0000000001);
    }

    [Fact]
    public void ReservoirRadius_Half_UsesSupportedWideEigensolver()
    {
        var layer = new ReservoirLayer<Half>(1, 2, connectionProbability: 0);
        var matrix = new Tensor<Half>(new[] { (Half)0, (Half)(-1000), (Half)1000, (Half)0 }, new[] { 2, 2 });
        Assert.Equal((Half)1000, layer.ComputeMaxEigenvalue(matrix));
    }

    [Fact]
    public void BicubicMatrix_MatchesKeysHalfPixelWeightsAndPreservesConstants()
    {
        var matrix = HtsatSpectrogramImageLayer<double>.BicubicMatrix(5, 3);
        Assert.Equal(1, matrix[0, 0], 12);
        Assert.Equal(0.5, matrix[1, 0], 12);
        Assert.Equal(0.59375, matrix[1, 1], 12);
        Assert.Equal(-0.09375, matrix[1, 2], 12);
        for (int row = 0; row < 5; row++)
            Assert.Equal(1, matrix[row, 0] + matrix[row, 1] + matrix[row, 2], 12);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void MixtureOfMamba_BatchedTopKRouting_ReturnsFiniteOutputs(int topK)
    {
        var layer = new MixtureOfMambaLayer<double>(3, 4, 3, topK, 2);
        var input = Tensor<double>.CreateDefault(new[] { 2, 3, 4 }, 0.25);
        var result = layer.Forward(input);
        Assert.Equal(input.Shape.ToArray(), result.Shape.ToArray());
        Assert.All(result.ToArray(), value => Assert.True(double.IsFinite(value)));
    }

    [Fact]
    public void Softcap_FloatWithWiderCap_PreservesFiniteSmallLogits()
    {
        var layer = new LogitSoftcapLayer<float>(1e40);
        var result = layer.Forward(new Tensor<float>(new[] { -2f, 0f, 2f }, new[] { 3 }));
        Assert.Equal(new[] { -2f, 0f, 2f }, result.ToArray());
    }

    [Fact]
    public void SsmQuantization_Half16Bits_LeavesRepresentableValuesFinite()
    {
        var method = typeof(SSMQuantizationHelper<Half>).GetMethod("QuantizeVector", BindingFlags.Static | BindingFlags.NonPublic)!;
        var parameters = new Vector<Half>(new[] { (Half)0, (Half)1, (Half)2 });
        var quantized = (Vector<Half>)method.Invoke(null, new object[] { parameters, 16 })!;
        Assert.Equal(parameters.ToArray(), quantized.ToArray());
    }

    [Fact]
    public void SelfAttention_HalfOneHotWeights_HaveFiniteAuxiliaryLoss()
    {
        var layer = new SelfAttentionLayer<Half>(2, 2, 1, activationFunction: null) { UseAuxiliaryLoss = true };
        layer.GetType().GetField("_lastAttentionScores", Private)!.SetValue(layer,
            new Tensor<Half>(new[] { (Half)1, (Half)0, (Half)0, (Half)1 }, new[] { 1, 1, 2, 2 }));
        Assert.True(Half.IsFinite(layer.ComputeAuxiliaryLoss()));
    }

    [Fact]
    public void Tps_HalfConstants_AreRoundedOnlyAfterWideGeometryAndInversion()
    {
        var narrow = new SVTRThinPlateSplineLayer<Half>(1, 32, 32, 3, 5, 4);
        var wide = new SVTRThinPlateSplineLayer<double>(1, 32, 32, 3, 5, 4);
        foreach (string name in new[] { "_inverseKernel", "_targetCoordinateRepresentation" })
        {
            var actual = (Tensor<Half>)narrow.GetType().GetField(name, Private)!.GetValue(narrow)!;
            var expected = (Tensor<double>)wide.GetType().GetField(name, Private)!.GetValue(wide)!;
            Assert.All(actual.ToArray(), value => Assert.True(Half.IsFinite(value)));
            Assert.Equal(expected.Cast<Half>().ToArray(), actual.ToArray());
        }
    }

    [Fact]
    public void DropPath_BatchedScale_IsUniformWithinEachSample()
    {
        var layer = new SVTRMixingBlockLayer<double>(4, 1, 1, 1, dropPathRate: 0.5);
        layer.SetTrainingMode(true);
        var method = layer.GetType().GetMethod("DropPath", Private)!;
        var result = (Tensor<double>)method.Invoke(layer,
            new object[] { Tensor<double>.CreateDefault(new[] { 2, 3 }, 1.0), 2 })!;
        for (int row = 0; row < 2; row++)
        {
            Assert.True(result[row, 0] == 0 || result[row, 0] == 2);
            Assert.Equal(result[row, 0], result[row, 1]);
            Assert.Equal(result[row, 0], result[row, 2]);
        }
    }
}
