using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests;

public class ElementwiseSweepRegressionTests
{
    [Fact]
    public void BandMultiply_SkippedPaddingDoesNotPropagateNonfiniteCoefficients()
    {
        var band = new Matrix<double>(new[,] { { double.NaN, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, double.NaN } });
        var result = new Vector<double>(new[] { double.NaN, double.NaN, double.NaN });
        MatrixHelper<double>.BandDiagonalMultiply(1, 1, band, result, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        Assert.Equal(new[] { 8.0, 32.0, 38.0 }, result.ToArray());
        Assert.True(double.IsNaN(band[0, 0]));
        Assert.True(double.IsNaN(band[2, 2]));
    }

    [Fact]
    public void HalfTabDDPM_ConstantNumericalColumnNormalizesToZero()
    {
        using var model = new TabDDPMGenerator<Half>(
            new NeuralNetworkArchitecture<Half>(inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new TabDDPMOptions<Half> { NumTimesteps = 2, NumCategoricalDiffusionSteps = 2,
                MLPDimensions = new[] { 4 }, TimestepEmbeddingDimension = 4, Seed = 7 });
        var columns = new[] { new ColumnMetadata("constant", ColumnDataType.Continuous) };
        var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        typeof(TabDDPMGenerator<Half>).GetMethod("AnalyzeColumns", flags)!.Invoke(model, new object[] { columns });
        var data = new Matrix<Half>(new[,] { { (Half)5 }, { (Half)5 }, { (Half)5 }, { (Half)5 } });
        var prepared = (ValueTuple<Matrix<Half>, Matrix<Half>>)typeof(TabDDPMGenerator<Half>)
            .GetMethod("PreprocessData", flags)!.Invoke(model, new object[] { data, columns })!;
        Assert.All(prepared.Item1.AsSpan().ToArray(), value => Assert.Equal((Half)0, value));
    }

    [Fact]
    public void HalfNormClipping_LargeFiniteVector_UsesWideAccumulation()
    {
        const int count = 70000;
        var input = new Vector<Half>(Enumerable.Repeat((Half)1, count).ToArray());
        var clipped = GradientClippingHelper.ClipByNorm(input, 10)!;
        double expected = 10 / Math.Sqrt(count);
        Assert.All(clipped.ToArray(), value => Assert.InRange((double)value, expected - 3e-5, expected + 3e-5));
        Assert.Equal(1.0, (double)input[0]);
    }

    [Fact]
    public void DoubleNormClipping_DoesNotSquareExtremeFiniteMagnitudes()
    {
        var input = new Vector<double>(new[] { 1e300, -1e300 });
        double expected = 1 / Math.Sqrt(2);
        var clipped = GradientClippingHelper.ClipByNorm(input, 1)!;
        Assert.Equal(expected, clipped[0], 12);
        Assert.Equal(-expected, clipped[1], 12);
        var global = GradientClippingHelper.ClipByGlobalNorm(
            new List<Vector<double>> { new(new[] { 1e300 }), new(new[] { -1e300 }) }, 1)!;
        Assert.Equal(expected, global[0][0], 12);
        Assert.Equal(-expected, global[1][0], 12);
    }

    [Fact]
    public void NormClipping_PreservesARepresentableSubnormalDirection()
    {
        var clipped = GradientClippingHelper.ClipByNorm(new Vector<float>(new[] { 1e-40f, -1e-40f }), 1e-42)!;
        Assert.InRange(clipped[0], 6.9e-43f, 7.2e-43f);
        Assert.InRange(clipped[1], -7.2e-43f, -6.9e-43f);
    }

    [Fact]
    public void HalfCausalMask_IsFiniteAndDoesNotReadFutureTokens()
    {
        using var decoder = new T5RelativeBiasAttentionLayer<Half>(4, 1, bidirectional: false, seed: 5);
        var input = new Tensor<Half>(
            Enumerable.Range(0, 12).Select(i => (Half)(0.01 * (i + 1))).ToArray(), new[] { 3, 4 });
        var full = decoder.Forward(input).ToArray();
        Assert.All(full, value => Assert.True(!Half.IsNaN(value) && !Half.IsInfinity(value)));
        var prefix = new Tensor<Half>(input.ToArray().Take(4).ToArray(), new[] { 1, 4 });
        Assert.Equal(decoder.Forward(prefix).ToArray(), full.Take(4).ToArray());
    }

    [Fact]
    public void MemoryUsagePenalty_UniformReadAndWriteWeightsHavePositiveShannonEntropy()
    {
        using var model = new NeuralTuringMachine<double> { UseAuxiliaryLoss = true };
        Assert.Equal(2 * Math.Log(64), model.ComputeAuxiliaryLoss(), 10);
    }

    [Fact]
    public void MemoryRead_SkippedLocationsDoNotPropagateNonfiniteValues()
    {
        using var model = new NeuralTuringMachine<double>();
        var memory = new Matrix<double>(64, 32);
        memory.AsWritableSpan().Fill(double.NaN);
        for (int column = 0; column < 32; column++) memory[0, column] = column + 1;
        var weights = new Vector<double>(64);
        weights[0] = 1;
        var read = typeof(NeuralTuringMachine<double>).GetMethod("ReadFromMemory",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        var result = (Vector<double>)read.Invoke(model, new object[] { memory, weights })!;
        Assert.Equal(Enumerable.Range(1, 32).Select(value => (double)value).ToArray(), result.ToArray());
    }

    [Fact]
    public void TabDDPMCheckpoint_RestoresFittedNormalizationAndReconstructionRange()
    {
        TabDDPMGenerator<double> Create(int seed) => new(
            new NeuralNetworkArchitecture<double>(inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            new TabDDPMOptions<double> { NumTimesteps = 2, NumCategoricalDiffusionSteps = 2,
                MLPDimensions = new[] { 4 }, TimestepEmbeddingDimension = 4, BatchSize = 4, Epochs = 1, Seed = seed });
        using var source = Create(7);
        using var restored = Create(19);
        var columns = new[] { new ColumnMetadata("value", ColumnDataType.Continuous) };
        source.Fit(new Matrix<double>(new[,] { { 10.0 }, { 20.0 }, { 30.0 }, { 40.0 } }), columns, epochs: 1);
        restored.Fit(new Matrix<double>(new[,] { { -40.0 }, { -30.0 }, { -20.0 }, { -10.0 } }), columns, epochs: 1);
        restored.Deserialize(source.Serialize());
        var generated = restored.Generate(3);
        Assert.Equal(3, generated.Rows);
        Assert.Equal(1, generated.Columns);
        Assert.All(generated.AsSpan().ToArray(), value => Assert.InRange(value, 10, 40));
    }

    [Fact]
    public void JackknifeMeanInterval_UsesOnlyThePairedPredictionPrefix()
    {
        var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
        var expected = StatisticsHelper<double>.CalculateJackknifeInterval(actual, new Vector<double>(new[] { 1.0, 2.0, 3.0 }));
        var extra = StatisticsHelper<double>.CalculateJackknifeInterval(actual, new Vector<double>(new[] { 1.0, 2.0, 3.0, 1e6 }));
        Assert.Equal(expected, extra);
    }

    [Fact]
    public void JackknifeMeanInterval_DoesNotLoseSpreadAtALargeCommonOffset()
    {
        const double offset = 1e16;
        var values = new Vector<double>(new[] { offset + 2, offset + 4, offset + 6, offset + 8, offset + 10 });
        var interval = StatisticsHelper<double>.CalculateJackknifeInterval(values, values);
        // Mean = offset + 6; s/sqrt(5) = sqrt(2), t(4,.975) ~= 2.776: endpoints round to +2 and +10.
        Assert.Equal(offset + 2, interval.Lower);
        Assert.Equal(offset + 10, interval.Upper);
    }
}
