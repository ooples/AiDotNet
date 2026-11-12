using AiDotNet.NeuralNetworks.Tabular;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Reference values for entmax (Peters, Niculae and Martins, 2019, "Sparse Sequence-to-Sequence Models"):
/// entmax_alpha(z) = [(alpha - 1) z - tau]_+^(1/(alpha-1)), tau chosen so the output sums to 1.
/// </summary>
public class EntmaxAttentionTests
{
    [Theory]
    [InlineData(1.5)]
    [InlineData(2.0)]
    public void LargeCommonOffset_PreservesUniformProbabilities(double alpha)
    {
        var output = new EntmaxAttention<double>(alpha).Forward(
            new Tensor<double>(new[] { 1e16, 1e16 }, new[] { 1, 2 }));
        Assert.Equal(0.5, output.GetFlat(0), 12);
        Assert.Equal(0.5, output.GetFlat(1), 12);
    }

    [Theory]
    [InlineData(1.5)]
    [InlineData(2.0)]
    public void ExtremeFiniteSpread_ConvergesToTheOneHotReference(double alpha)
    {
        var output = new EntmaxAttention<double>(alpha).Forward(
            new Tensor<double>(new[] { 1e308, -1e308, -1e308 }, new[] { 1, 3 }));
        Assert.Equal(1.0, output.GetFlat(0), 12);
        Assert.Equal(0.0, output.GetFlat(1), 12);
        Assert.Equal(0.0, output.GetFlat(2), 12);
    }

    [Theory]
    [InlineData(1.0001)]
    [InlineData(1.5)]
    [InlineData(2.0)]
    public void HalfThresholdGeometry_PreservesUniformProbabilities(double alpha)
    {
        var output = new EntmaxAttention<System.Half>(alpha).Forward(
            new Tensor<System.Half>(new[] { (System.Half)10000, (System.Half)10000 }, new[] { 1, 2 }));
        Assert.Equal((System.Half)0.5, output.GetFlat(0));
        Assert.Equal((System.Half)0.5, output.GetFlat(1));
    }

    [Fact]
    public void Entmax15_MatchesThePaperClosedForm()
    {
        // z = [1, 0], alpha = 1.5: both entries are in the support, so (0.5 - tau)^2 + (-tau)^2 = 1,
        // tau = (1 - sqrt 7) / 4 and p = [(0.5 - tau)^2, tau^2] = [0.830719, 0.169281]. Dropping the
        // (alpha - 1) scale (the old code) gives tau = 0 and the degenerate [1, 0].
        var entmax = new EntmaxAttention<double>(1.5);
        var output = entmax.Forward(new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 1.0, 0.0 })));

        double tau = (1.0 - System.Math.Sqrt(7.0)) / 4.0;
        Assert.Equal((0.5 - tau) * (0.5 - tau), output[0, 0], 6);
        Assert.Equal(tau * tau, output[0, 1], 6);
    }

    [Fact]
    public void Sparsemax_MatchesThePaperClosedForm()
    {
        // alpha = 2 is sparsemax (Martins and Astudillo, 2016): z = [1, 0.5, -1] gives tau = 0.25,
        // p = [0.75, 0.25, 0].
        var sparsemax = new EntmaxAttention<double>(2.0);
        var output = sparsemax.Forward(new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { 1.0, 0.5, -1.0 })));

        Assert.Equal(0.75, output[0, 0], 10);
        Assert.Equal(0.25, output[0, 1], 10);
        Assert.Equal(0.0, output[0, 2], 10);
    }
}
