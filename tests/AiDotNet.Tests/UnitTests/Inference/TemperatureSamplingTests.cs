using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// <see cref="TemperatureSampling{T}"/> runs on engine ops (relu, power, sum, cumsum, searchsorted). These tests pin
/// it to the scalar definitions it replaced: <c>p_i^(1/t)</c> over the positive entries renormalized, and the
/// inverse-CDF draw that skips entries without mass.
/// </summary>
public class TemperatureSamplingTests
{
    private static double[] ReferenceApply(double[] p, double t)
    {
        double max = 0;
        foreach (double v in p) if (v > max) max = v;
        var result = new double[p.Length];
        double sum = 0;
        for (int i = 0; i < p.Length; i++)
        {
            result[i] = p[i] > 0 ? Math.Pow(p[i] / max, 1.0 / t) : 0;
            sum += result[i];
        }
        for (int i = 0; i < p.Length; i++) result[i] /= sum;
        return result;
    }

    private static int ReferenceSample(double[] p, double r)
    {
        double cumulative = 0;
        int lastNonZero = -1;
        for (int i = 0; i < p.Length; i++)
        {
            if (!(p[i] > 0)) continue;
            lastNonZero = i;
            cumulative += p[i];
            if (r < cumulative) return i;
        }
        return lastNonZero;
    }

    private static double[] RandomDistribution(int length, int seed, bool withZeros)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var p = new double[length];
        double sum = 0;
        for (int i = 0; i < length; i++)
        {
            p[i] = withZeros && i % 3 == 1 ? 0 : random.NextDouble();
            sum += p[i];
        }
        for (int i = 0; i < length; i++) p[i] /= sum;
        return p;
    }

    [Theory]
    [InlineData(0.25)]
    [InlineData(0.7)]
    [InlineData(1.6)]
    public void Apply_MatchesTheScalarDefinition_WithZeroAndNegativeEntries(double temperature)
    {
        var p = RandomDistribution(257, 11, withZeros: true);
        p[5] = -0.01;
        var expected = ReferenceApply(p, temperature);

        var actual = TemperatureSampling<double>.Apply(new Vector<double>(p), temperature);

        for (int i = 0; i < p.Length; i++) Assert.Equal(expected[i], actual[i], 12);
        Assert.Equal(0.0, actual[5]);
        Assert.Equal(0.0, actual[1]);
    }

    [Fact]
    public void Apply_TemperatureOne_ReturnsACopy()
    {
        var input = new Vector<double>(new[] { 0.2, 0.3, 0.5 });

        var result = TemperatureSampling<double>.Apply(input, 1.0);
        result[0] = 9;

        Assert.Equal(0.2, input[0]);
    }

    [Fact]
    public void Apply_ExtremeLowTemperature_IsTheGreedyOneHot()
    {
        var input = new Vector<double>(new[] { 1e-300, 2e-300, 1.5e-300 });

        var result = TemperatureSampling<double>.Apply(input, 1e-300);

        Assert.Equal(new[] { 0.0, 1.0, 0.0 }, result.ToArray());
    }

    [Fact]
    public void ArgMax_ReturnsTheFirstIndexOnTies()
    {
        Assert.Equal(1, TemperatureSampling<double>.ArgMax(new Vector<double>(new[] { 0.1, 0.4, 0.1, 0.4 })));
    }

    [Fact]
    public void Sample_DrawsTheSameIndexAsTheScalarInverseCdf()
    {
        var p = RandomDistribution(97, 23, withZeros: true);
        var vector = new Vector<double>(p);
        var engineRandom = RandomHelper.CreateSeededRandom(5);
        var referenceRandom = RandomHelper.CreateSeededRandom(5);

        for (int draw = 0; draw < 2000; draw++)
        {
            int actual = TemperatureSampling<double>.Sample(vector, engineRandom);
            Assert.Equal(ReferenceSample(p, referenceRandom.NextDouble()), actual);
            Assert.True(p[actual] > 0, $"Draw {draw} chose index {actual}, which has no mass.");
        }
    }

    [Fact]
    public void Sample_OneHot_AlwaysReturnsTheHotIndex_EvenWhenItIsFirst()
    {
        var random = RandomHelper.CreateSeededRandom(3);
        foreach (int hot in new[] { 0, 4, 9 })
        {
            var oneHot = Vector<float>.CreateStandardBasis(10, hot);
            for (int draw = 0; draw < 200; draw++) Assert.Equal(hot, TemperatureSampling<float>.Sample(oneHot, random));
        }
    }

    [Fact]
    public void Sample_MassJustBelowTheDraw_ReturnsTheLastIndexWithMass()
    {
        // A total of 0.5 leaves every draw at or above it past the end of the cumulative sum.
        var p = new Vector<double>(new[] { 0.25, 0.25, 0.0, 0.0 });
        var random = new FixedRandom(0.75);

        Assert.Equal(1, TemperatureSampling<double>.Sample(p, random));
    }

    private sealed class FixedRandom : Random
    {
        private readonly double _value;
        public FixedRandom(double value) => _value = value;
        public override double NextDouble() => _value;
    }
}
