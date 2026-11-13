using System;
using System.Linq;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.Latest;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The samplers' tie order: equal scores are visited in ascending index order (LINQ's <c>ThenBy(i)</c> / stable
/// <c>OrderBy</c> in the reference samplers). Runs of 64 equal values are long enough that the engine's unstable
/// argsort scrambles them, so a sampler that dropped the int tie pass would land on a different token.
/// </summary>
public class TtsSamplingTieTests
{
    private const int Ties = 64;

    /// <summary>A random source whose doubles are fixed, so a draw lands on a chosen position of the visiting order.</summary>
    internal sealed class FixedRandom : Random
    {
        private readonly double _value;
        public FixedRandom(double value) { _value = value; }
        public override double NextDouble() => _value;
        protected override double Sample() => _value;
    }

    // A draw u = (j + ½)/n over n equal weights lands on the j-th visited one.
    private static double Middle(int j, int n) => (j + .5) / n;

    [Fact]
    public void StableOrder_PutsEqualValuesInAscendingIndexOrder()
    {
        var values = new double[2 * Ties];
        for (int i = 0; i < values.Length; i++) values[i] = i % 2 == 0 ? 1 : 2;
        var tensor = new Tensor<double>(values, new[] { values.Length });
        var expectedDescending = Enumerable.Range(0, values.Length).OrderByDescending(i => values[i]).ThenBy(i => i).ToArray();
        var expectedAscending = Enumerable.Range(0, values.Length).OrderBy(i => values[i]).ThenBy(i => i).ToArray();
        Assert.Equal(expectedDescending, TtsSampling<double>.StableOrder(tensor, descending: true));
        Assert.Equal(expectedAscending, TtsSampling<double>.StableOrder(tensor, descending: false));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(17)]
    [InlineData(40)]
    [InlineData(Ties - 1)]
    public void MegaTtsTopK_VisitsTiedLogitsInIndexOrder(int j)
    {
        var logits = new Tensor<double>(Enumerable.Repeat(.5, Ties).ToArray(), new[] { Ties });
        Assert.Equal(j, MegaTTS<double>.TopK(logits, Ties, new FixedRandom(Middle(j, Ties))));
        var floatLogits = new Tensor<float>(Enumerable.Repeat(.5f, Ties).ToArray(), new[] { Ties });
        Assert.Equal(j, MegaTTS<float>.TopK(floatLogits, Ties, new FixedRandom(Middle(j, Ties))));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(5)]
    [InlineData(33)]
    [InlineData(Ties - 1)]
    public void MegaTts2TopK_VisitsTiedProbabilitiesInIndexOrder(int j)
    {
        // Index 0 (the start token) carries no mass; the 64 codes after it tie.
        var p = new double[Ties + 1];
        for (int c = 1; c <= Ties; c++) p[c] = 1.0 / Ties;
        Assert.Equal(j + 1, MegaTTS2<double>.TopK(new Tensor<double>(p, new[] { p.Length }), Ties, new FixedRandom(Middle(j, Ties))));
    }

    [Fact]
    public void MegaTts2TopK_KeepsTheLowestIndicesAmongTiesAtTheCut()
    {
        // k = 4 of 64 tied codes: master keeps codes 1..4, so every draw lands in them.
        var p = new double[Ties + 1];
        for (int c = 1; c <= Ties; c++) p[c] = 1.0 / Ties;
        var tensor = new Tensor<double>(p, new[] { p.Length });
        for (int j = 0; j < 4; j++)
            Assert.Equal(j + 1, MegaTTS2<double>.TopK(tensor, 4, new FixedRandom(Middle(j, 4))));
    }
}
