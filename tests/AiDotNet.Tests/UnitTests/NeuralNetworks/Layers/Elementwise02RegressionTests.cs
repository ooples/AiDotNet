using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

public class Elementwise02RegressionTests
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;

    private static MethodInfo FindMethod(Type type, string name, BindingFlags flags, Type[]? parameters = null)
    {
        var method = parameters is null
            ? type.GetMethod(name, flags)
            : type.GetMethod(name, flags, null, parameters, null);
        return method ?? throw new InvalidOperationException($"{type.Name}.{name} was not found.");
    }

    private static TResult Invoke<TResult>(MethodInfo method, object? target, params object[] arguments)
    {
        object? result = method.Invoke(target, arguments);
        return result is TResult typed
            ? typed
            : throw new InvalidOperationException($"{method.Name} did not return a {typeof(TResult).Name}.");
    }

    private static FieldInfo FindField(Type type, string name)
        => type.GetField(name, Private) ?? throw new InvalidOperationException($"{type.Name}.{name} was not found.");

    private static TValue ReadField<TValue>(object owner, string name)
    {
        object? value = FindField(owner.GetType(), name).GetValue(owner);
        return value is TValue typed
            ? typed
            : throw new InvalidOperationException($"{owner.GetType().Name}.{name} is not a {typeof(TValue).Name}.");
    }

    // double.IsFinite and Half.IsFinite do not exist on net471.
    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);

    private static bool IsFinite(Half value) => IsFinite((double)(float)value);

    [Fact]
    public void RbfSpread_LargeHighDimensionalCloud_MatchesClosedForm()
    {
        const int count = 512, dimensions = 768;
        var layer = new RBFLayer<double>(1, 2);
        var centers = new Tensor<double>(new[] { count, dimensions });
        for (int i = (count - 1) * dimensions; i < centers.Length; i++) centers.SetFlat(i, 1.0);
        var method = FindMethod(layer.GetType(), "CenterSpreadWidth", Private,
            new[] { typeof(Tensor<double>), typeof(int) });
        double width = Invoke<double>(method, layer, centers, count);
        // The most distant pair differs by 1 in each dimension: d_max^2 = dimensions.
        Assert.Equal(Math.Sqrt(dimensions / (2.0 * count)), width, 12);
    }

    [Fact]
    public void RbfSpread_HalfLargeCoordinates_UsesWideSquaredDistances()
    {
        var layer = new RBFLayer<Half>(1, 2);
        var centers = new Tensor<Half>(new[] { (Half)(-1000), (Half)1000 }, new[] { 2, 1 });
        var method = FindMethod(layer.GetType(), "CenterSpreadWidth", Private,
            new[] { typeof(Tensor<Half>), typeof(int) });
        Assert.Equal((Half)1000, Invoke<Half>(method, layer, centers, 2));
    }

    [Fact]
    public void MeshSort_LargeEdgeSet_UsesAscendingStableOrder()
    {
        const int count = 16384;
        var layer = new MeshPoolLayer<double>(1, 1);
        var values = new Tensor<double>(Enumerable.Range(0, count).Select(i => (double)(count - i)).ToArray(), new[] { count });
        var method = FindMethod(layer.GetType(), "SortEdgesByImportance", Private);
        var order = Invoke<int[]>(method, layer, values, count);
        Assert.Equal(Enumerable.Range(0, count).Reverse(), order);
        var tied = new Tensor<double>(new[] { 2.0, double.NaN, 1.0, 1.0, double.NaN, 2.0 }, new[] { 6 });
        Assert.Equal(new[] { 1, 4, 2, 3, 0, 5 }, Invoke<int[]>(method, layer, tied, 6));
    }

    [Fact]
    public void HalfCausalBias_DistinguishesAdjacentKeysBeyond2048()
    {
        var layer = new MultiHeadAttentionLayer<Half>(1, 2) { UseCausalMask = true };
        var method = FindMethod(layer.GetType(), "AttentionDropoutLogitBias", Private);
        var bias = Invoke<Tensor<Half>>(method, layer, 1, 2, 2050);
        Assert.Equal((Half)0, bias[0, 0, 2048]);
        Assert.True(Half.IsNegativeInfinity(bias[0, 0, 2049]));
        Assert.Equal((Half)0, bias[0, 1, 2049]);
    }

    [Fact]
    public void HalfLayoutCoordinates_HandleNanAndRoundTiesToEven()
    {
        var layer = new LayoutEmbeddingLayer<Half>(vocabSize: 4, hiddenDim: 2,
            maxSequenceLength: 4, maxPosition2D: 8, includeTokens: false);
        var method = FindMethod(layer.GetType(), "ClampCoordinate", Private);
        var input = new Tensor<Half>(new[] { Half.NaN, (Half)2.5, (Half)3.5, (Half)(-1), (Half)20 }, new[] { 5 });
        var result = Invoke<Tensor<Half>>(method, layer, input);
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

    // Keys' cubic cancels terms near 3 down to weights near 0.1, so evaluating it in Half cost several
    // ulps per weight (row 1 of the 4x3 matrix came out 0.3120 instead of 0.3147). The reference forms each
    // weight in double and rounds once.
    [Theory]
    [InlineData(4, 3)]
    [InlineData(7, 3)]
    [InlineData(256, 64)]
    public void BicubicMatrix_Half_RoundsDoubleWeightsOnce(int outputs, int inputs)
    {
        var narrow = HtsatSpectrogramImageLayer<Half>.BicubicMatrix(outputs, inputs);
        var wide = HtsatSpectrogramImageLayer<double>.BicubicMatrix(outputs, inputs);
        Assert.Equal(wide.Cast<Half>().ToArray(), narrow.ToArray());
    }

    // RWKV-v7 reference init: w0 = -6 + 6 n/(C-1) + 0.5 + 2.5 zigzag, k_k = 0.71 - 0.1 linear,
    // v0 = 0.73 - 0.4 linear, x_g = 1 - (n/C)^0.2, each evaluated in double and rounded to T once.
    [Fact]
    public void Rwkv7_FloatInitRamps_AreRoundedOnceFromDouble()
    {
        var narrow = new RWKV7Block<float>(4, 16, 4);
        var wide = new RWKV7Block<double>(4, 16, 4);
        foreach (string name in new[] { "_aBias", "_kk", "_v0", "_timeMixG" })
        {
            var expected = ReadField<Tensor<double>>(wide, name).Cast<float>().ToArray();
            Assert.Equal(expected, ReadField<Tensor<float>>(narrow, name).ToArray());
        }
    }

    // S4D-Lin: Im(A_n) = pi (n + 1). Rounding pi to float before the product is off by an ulp at n = 4.
    [Fact]
    public void S4dAndS5_FloatFrequencies_AreRoundedOnceFromDouble()
    {
        var s4dNarrow = new S4DLayer<float>(4, 4, 16);
        var s4dWide = new S4DLayer<double>(4, 4, 16);
        Assert.Equal(ReadField<Tensor<double>>(s4dWide, "_aImag").Cast<float>().ToArray(),
            ReadField<Tensor<float>>(s4dNarrow, "_aImag").ToArray());

        var s5Narrow = new S5Layer<float>(4, 4, 16);
        var s5Wide = new S5Layer<double>(4, 4, 16);
        Assert.Equal(ReadField<Tensor<double>>(s5Wide, "_aImag").Cast<float>().ToArray(),
            ReadField<Tensor<float>>(s5Narrow, "_aImag").ToArray());
    }

    // In double the S4D-Lin frequencies are exactly n * Math.PI, formed by plain double arithmetic rather than an
    // engine op a GPU engine could evaluate in float.
    [Fact]
    public void S4d_DoubleFrequencies_AreExactMultiplesOfPi()
    {
        var layer = new S4DLayer<double>(4, 4, 16);
        var aImag = ReadField<Tensor<double>>(layer, "_aImag").ToArray();
        var expected = Enumerable.Range(1, 16).Select(n => n * Math.PI).ToArray();
        Assert.Equal(expected, aImag.Take(16).ToArray());
    }

    // The Born-rule division ran before the zero-row selection; decimal division by zero throws, so an all-zero
    // row must divide by a safe divisor and still come out as zeros.
    [Fact]
    public void Measurement_DecimalZeroRow_ReturnsZeros()
    {
        var layer = new MeasurementLayer<decimal>(4);
        var output = layer.Forward(new Tensor<decimal>(new[] { 2, 4 }));
        Assert.All(output.ToArray(), p => Assert.Equal(0m, p));
    }

    // gamma_h = clamp(decay + (h - (H-1)/2) / (H-1) * 0.02, 0.9, 0.9999); in float the three roundings put
    // head 0 of a 4-head, 0.97 layer at 0.96000004 instead of 0.96.
    [Fact]
    public void TransNormer_FloatDecays_AreRoundedOnceFromDouble()
    {
        var narrow = new TransNormerLLMLayer<float>(4, 16, 4, 0.97);
        var wide = new TransNormerLLMLayer<double>(4, 16, 4, 0.97);
        Assert.Equal(ReadField<Tensor<double>>(wide, "_gammas").Cast<float>().ToArray(),
            ReadField<Tensor<float>>(narrow, "_gammas").ToArray());
    }

    // Align-corners grid x = w * 2 / (W - 1) - 1: in float the product and the "- 1" each round, so
    // w = 1 of a 7-wide grid lands an ulp off -2/3.
    [Theory]
    [InlineData(5, 7)]
    [InlineData(7, 9)]
    public void SpyNetIdentityGrid_Float_IsRoundedOnceFromDouble(int height, int width)
    {
        var narrow = SpyNetLayer<float>.IdentityGrid(height, width);
        var wide = SpyNetLayer<double>.IdentityGrid(height, width);
        Assert.Equal(new[] { height, width, 2 }, narrow.Shape.ToArray());
        Assert.Equal(wide.Cast<float>().ToArray(), narrow.ToArray());
        for (int w = 0; w < width; w++)
            Assert.Equal((float)(w * (2.0 / (width - 1)) - 1.0), narrow[0, w, 0]);
    }

    // Izhikevich (2003) step from v = -300, u = 0 with I = -2000:
    // v' = v + 0.04 v^2 + 5 v + 140 - u + I = -300 + 3600 - 1500 + 140 - 2000 = -60, below the 30 mV peak.
    // v * v = 90000 overflows Half, which turned this into an infinite potential and a spurious spike.
    [Fact]
    public void Izhikevich_HalfStrongHyperpolarization_StaysFiniteWithoutSpike()
    {
        var layer = new SpikingLayer<Half>(1, 1, SpikingNeuronType.Izhikevich);
        FindField(layer.GetType(), "_membranePotential").SetValue(layer, new Tensor<Half>(new[] { (Half)(-300) }, new[] { 1 }));
        FindField(layer.GetType(), "_recoveryVariable").SetValue(layer, new Tensor<Half>(new[] { (Half)0 }, new[] { 1 }));
        var method = FindMethod(layer.GetType(), "UpdateIzhikevich", Private);

        var spikes = Invoke<Tensor<Half>>(method, layer, new Tensor<Half>(new[] { (Half)(-2000) }, new[] { 1 }));

        Assert.Equal((Half)0, spikes[0]);
        Assert.Equal((Half)(-60), ReadField<Tensor<Half>>(layer, "_membranePotential")[0]);
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
        Assert.All(result.ToArray(), value => Assert.True(IsFinite(value)));
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
        var method = FindMethod(typeof(SSMQuantizationHelper<Half>), "QuantizeVector", BindingFlags.Static | BindingFlags.NonPublic);
        var parameters = new Vector<Half>(new[] { (Half)0, (Half)1, (Half)2 });
        var quantized = Invoke<Vector<Half>>(method, null, parameters, 16);
        Assert.Equal(parameters.ToArray(), quantized.ToArray());
    }

    [Fact]
    public void SelfAttention_HalfOneHotWeights_HaveFiniteAuxiliaryLoss()
    {
        var layer = new SelfAttentionLayer<Half>(2, 2, 1, activationFunction: null) { UseAuxiliaryLoss = true };
        FindField(layer.GetType(), "_lastAttentionScores").SetValue(layer,
            new Tensor<Half>(new[] { (Half)1, (Half)0, (Half)0, (Half)1 }, new[] { 1, 1, 2, 2 }));
        Assert.True(IsFinite(layer.ComputeAuxiliaryLoss()));
    }

    [Fact]
    public void Tps_HalfConstants_AreRoundedOnlyAfterWideGeometryAndInversion()
    {
        var narrow = new SVTRThinPlateSplineLayer<Half>(1, 32, 32, 3, 5, 4);
        var wide = new SVTRThinPlateSplineLayer<double>(1, 32, 32, 3, 5, 4);
        foreach (string name in new[] { "_inverseKernel", "_targetCoordinateRepresentation" })
        {
            var actual = ReadField<Tensor<Half>>(narrow, name);
            var expected = ReadField<Tensor<double>>(wide, name);
            Assert.All(actual.ToArray(), value => Assert.True(IsFinite(value)));
            Assert.Equal(expected.Cast<Half>().ToArray(), actual.ToArray());
        }
    }

    [Fact]
    public void DropPath_BatchedScale_IsUniformWithinEachSample()
    {
        var layer = new SVTRMixingBlockLayer<double>(4, 1, 1, 1, dropPathRate: 0.5);
        layer.SetTrainingMode(true);
        var method = FindMethod(layer.GetType(), "DropPath", Private);
        var result = Invoke<Tensor<double>>(method, layer, Tensor<double>.CreateDefault(new[] { 2, 3 }, 1.0), 2);
        for (int row = 0; row < 2; row++)
        {
            Assert.True(result[row, 0] == 0 || result[row, 0] == 2);
            Assert.Equal(result[row, 0], result[row, 1]);
            Assert.Equal(result[row, 0], result[row, 2]);
        }
    }
}
