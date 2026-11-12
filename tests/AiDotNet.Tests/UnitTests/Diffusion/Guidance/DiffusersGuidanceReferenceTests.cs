using System;
using System.Threading.Tasks;
using AiDotNet.Diffusion.Guidance;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Guidance;

/// <summary>
/// Adaptive projected guidance and rescaled CFG reproduce diffusers 0.35
/// (<c>tools/reference-data/diffusers_guidance_reference.py</c>) on a batch of two samples.
/// </summary>
/// <remarks>
/// Before this, APG scaled the orthogonal component and kept the parallel one (the paper keeps the orthogonal
/// component and down-weights the oversaturating parallel one), and both methods pooled their projection and
/// standard deviations over the whole batch instead of per sample.
/// </remarks>
public class DiffusersGuidanceReferenceTests
{
    private const double Tolerance = 1e-12;

    private static Tensor<double> Batch(double[] values) => new(new[] { 2, 1, 2, 3 }, new Vector<double>(values));

    private static readonly double[] Unconditional = { 0.1, -0.4, 0.7, 0.2, -0.3, 0.5, 1.2, -0.8, 0.3, 0.0, 0.6, -1.1 };
    private static readonly double[] Conditional = { 0.3, -0.1, 0.9, -0.2, 0.1, 0.4, 0.7, -0.2, 0.8, 0.5, 0.1, -0.9 };

    private static void AssertClose(double[] expected, Tensor<double> actual)
    {
        var values = actual.ToArray();
        Assert.Equal(expected.Length, values.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - values[i]) <= Tolerance, $"[{i}] expected {expected[i]:R}, got {values[i]:R}");
    }

    [Fact(Timeout = 120000)]
    public async Task AdaptiveProjectedGuidance_MatchesDiffusersNormalizedGuidance()
    {
        await Task.Yield();
        var guidance = new AdaptiveProjectedGuidance<double>(projectionStrength: 0.5);
        var result = guidance.Apply(Batch(Unconditional), Batch(Conditional), 7.5, 0.5);
        AssertClose(new[]
        {
            1.30870535714286, 1.94709821428571, 1.32611607142857, -2.60580357142857, 2.60290178571429, -0.638392857142857, -2.49140625, 3.68325892857143, 4.11696428571429, 3.79185267857143, -3.14162946428571, 0.324665178571429,
        }, result);
    }

    [Fact(Timeout = 120000)]
    public async Task RescaledCFG_MatchesDiffusersRescaleNoiseCfg()
    {
        await Task.Yield();
        var guidance = new RescaledCFG<double>(rescaleWeight: 0.7);
        var result = guidance.Apply(Batch(Unconditional), Batch(Conditional), 7.5, 0.5);
        AssertClose(new[]
        {
            0.695981480687581, 0.804728587045016, 0.956974535945424, -1.21796759120327, 1.17446874866029, -0.108747106357434, -1.11369009479563, 1.61594249048778, 1.76880191526366, 1.63777955117005, -1.37573482298284, 0.174696485458139,
        }, result);
    }
}
