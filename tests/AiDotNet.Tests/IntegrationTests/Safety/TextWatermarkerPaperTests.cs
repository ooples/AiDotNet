using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

public class TextWatermarkerPaperTests
{
    // Kirchenbauer et al. (2023), Eq. (2): z = (|s|_G - gamma T) / sqrt(T gamma (1-gamma)).
    [Theory]
    [InlineData(0.25, 12, 4.618802153517006)]
    [InlineData(0.5, 12, 2.0)]
    [InlineData(0.75, 8, -2.309401076758503)]
    public void DetectionMatchesThePapersBinomialStatistic(double gamma, int greenCount, double expectedZ)
    {
        var marker = new TextWatermarker<double>(key: "paper-equation-test", greenListFraction: gamma);
        var tokens = new List<int> { 0 };
        for (int i = 0; i < 16; i++)
        {
            bool wantGreen = i < greenCount;
            // Disjoint token ranges ensure every scored bigram is distinct.
            int token = Enumerable.Range(1 + i * 10000, 10000)
                .First(candidate => marker.IsGreen(tokens, candidate) == wantGreen);
            tokens.Add(token);
        }

        var result = marker.DetectTokens(tokens);
        Assert.Equal(16, result.ScoredUnits);
        Assert.Equal(expectedZ, result.ZScore, 12);
    }

    [Fact]
    public void SoftWatermarkAddsDeltaOnlyToGreenLogits()
    {
        // Algorithm 2 adds delta to green logits; default strength 0.5 corresponds to delta = 2.
        var marker = new TextWatermarker<double>(key: "paper-equation-test");
        int[] context = [7];
        var original = Enumerable.Range(0, 64).Select(i => i / 10.0 - 3.0).ToArray();
        var logits = new Vector<double>(original.ToArray());
        marker.BiasLogits(logits, context);

        Assert.Contains(Enumerable.Range(0, 64), i => marker.IsGreen(context, i));
        Assert.Contains(Enumerable.Range(0, 64), i => !marker.IsGreen(context, i));
        for (int i = 0; i < logits.Length; i++)
            Assert.Equal(original[i] + (marker.IsGreen(context, i) ? 2.0 : 0.0), logits[i], 12);
    }
}
