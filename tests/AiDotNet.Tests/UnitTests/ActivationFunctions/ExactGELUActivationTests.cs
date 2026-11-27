using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ActivationFunctions;

/// <summary>
/// The exact GELU against reference values of x·Φ(x), and its derivative Φ(x) + x·φ(x); and the encoder block's
/// LayerNorm epsilon surviving a save.
/// </summary>
public sealed class ExactGELUActivationTests
{
    // x·Φ(x) and Φ(x) + x·φ(x), from Python's math.erfc and the normal density in double precision.
    [Theory]
    [InlineData(-3.0, -0.004049694094890288, -0.011945647204183927)]
    [InlineData(-1.0, -0.15865525393145707, -0.08331547058768629)]
    [InlineData(0.0, 0.0, 0.5)]
    [InlineData(0.5, 0.3457312306370065, 0.8674951246561627)]
    [InlineData(1.0, 0.8413447460685428, 1.0833154705876862)]
    [InlineData(2.0, 1.9544997361036418, 1.085231801078197)]
    public void MatchesTheExactFunction(double x, double value, double derivative)
    {
        var gelu = new ExactGELUActivation<double>();

        Assert.Equal(value, gelu.Activate(x), 9);
        Assert.Equal(derivative, gelu.Derivative(x), 9);

        var tensor = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { x }));
        Assert.Equal(value, gelu.Activate(tensor)[0], 9);
        Assert.Equal(derivative, gelu.Derivative(tensor)[0], 9);
    }

    [Fact]
    public void DiffersFromTheTanhApproximation()
    {
        // The tanh form is off by about 1e-4 at x = -2 (4e-4 at x = -3); a model that must reproduce an erf-GELU
        // checkpoint cannot use it.
        double exact = new ExactGELUActivation<double>().Activate(-2.0);
        double approximate = new GELUActivation<double>().Activate(-2.0);

        Assert.True(Math.Abs(exact - approximate) > 1e-5, $"exact {exact} vs tanh {approximate}");
    }

    [Fact]
    public void EncoderBlock_KeepsItsLayerNormEpsilonThroughMetadata()
    {
        var block = new TransformerEncoderBlock<double>(8, 2, 16, 0.0, new ExactGELUActivation<double>(),
            TransformerNormPlacement.PostNorm, layerNormEpsilon: 1e-12);

        var metadata = block.GetMetadata();

        Assert.Equal(1e-12, block.LayerNormEpsilon);
        Assert.Equal(1e-12, double.Parse(metadata["LayerNormEpsilon"], System.Globalization.CultureInfo.InvariantCulture));
    }

    [Fact]
    public void EncoderBlock_RefusesANonPositiveEpsilon()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TransformerEncoderBlock<double>(
            8, 2, 16, layerNormEpsilon: 0.0));
    }
}
