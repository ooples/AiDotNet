using System;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Piecewise linear encoding against Gorishniy, Rubachev and Babenko (2022): e_t = (x - b_{t-1}) / (b_t - b_{t-1})
/// clamped to [0, 1], except that the first bin is not clamped below and the last is not clamped above.
/// </summary>
public class PiecewiseLinearEncodingLayerTests
{
    // Default edges for 4 bins: -2, -1, 0, 1, 2 (each bin has width 1).
    private static PiecewiseLinearEncodingLayer<double> Layer() => new(numFeatures: 1, numBins: 4);

    private static double[] Encode(PiecewiseLinearEncodingLayer<double> layer, double x)
    {
        var input = new Tensor<double>([1, 1]);
        input[0] = x;
        return layer.Forward(input).ToArray();
    }

    [Theory]
    [InlineData(0.5, new[] { 1.0, 1.0, 0.5, 0.0 })]    // inside bin 3
    [InlineData(-1.5, new[] { 0.5, 0.0, 0.0, 0.0 })]   // inside bin 1
    [InlineData(-3.0, new[] { -1.0, 0.0, 0.0, 0.0 })]  // below every edge: the first bin is not clamped below
    [InlineData(3.0, new[] { 1.0, 1.0, 1.0, 2.0 })]    // above every edge: the last bin is not clamped above
    public void Encoding_MatchesThePaper(double x, double[] expected)
    {
        var encoded = Encode(Layer(), x);
        Assert.Equal(expected.Length, encoded.Length);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], encoded[i], 12);
    }

    [Fact]
    public void Gradient_ReachesTheInput_WithTheActiveBinsSlope()
    {
        // d/dx sum_t e_t at x = 0.5 is 1 / width of the one bin x is inside (bin 3, width 1); the clamped bins add 0.
        var layer = Layer();
        var input = new Tensor<double>([1, 1]);
        input[0] = 0.5;
        using var tape = new GradientTape<double>();
        var output = layer.Forward(input);
        var loss = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.ReduceSum(output, null);
        var grads = tape.ComputeGradients(loss, new[] { input });
        Assert.True(grads.TryGetValue(input, out var g) && g is not null, "no gradient reached the input");
        Assert.Equal(1.0, g![0], 9);
    }

    [Fact]
    public void FitBoundaries_UsesTheDataQuantiles_AndKeepsEveryBinPositive()
    {
        var layer = new PiecewiseLinearEncodingLayer<double>(numFeatures: 2, numBins: 2);
        var data = new Tensor<double>([5, 2]);
        double[] first = { 0, 1, 2, 3, 4 };
        for (int s = 0; s < 5; s++)
        {
            data[s * 2] = first[s];
            data[s * 2 + 1] = 7.0;   // constant feature: every quantile ties
        }
        layer.FitBoundaries(data);

        // Feature 0: quantiles at 0, 1/2, 1 are 0, 2, 4, so x = 1 sits halfway through bin 1 and has not reached bin 2.
        var probe = new Tensor<double>([1, 2]);
        probe[0] = 1.0;
        probe[1] = 7.0;
        var encoded = layer.Forward(probe).ToArray();
        Assert.Equal(0.5, encoded[0], 9);
        Assert.Equal(0.0, encoded[1], 9);
        Assert.All(encoded, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v)));
    }

    [Fact]
    public void TheEdgesArePersistedState_NotTrainableParameters()
    {
        // Fixed edges, as in the paper: saved with the model (a [Buffer], counted by ParameterCount for persistence)
        // but never handed to the optimizer or the tape.
        var layer = Layer();
        Assert.Empty(layer.GetTrainableParameters());
        Assert.Equal(5, layer.ParameterCount);   // 1 feature x (4 bins + 1) edges
    }
}

/// <summary>The layer in the invariant harness (finite output, replay, shape, serialization, input gradients).</summary>
public sealed class PiecewiseLinearEncodingLayerInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 3];
    protected override ILayer<double> CreateLayer() => new PiecewiseLinearEncodingLayer<double>(numFeatures: 3, numBins: 4);
}
