using System;
using AiDotNet.Metrics;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Metrics;

/// <summary>
/// Unit tests for <see cref="LanguageModelMetrics{T}"/> — perplexity, cross-entropy, and top-k
/// accuracy over a vocabulary. Values are checked against hand-computed references.
/// </summary>
public class LanguageModelMetricsTests
{
    private static Tensor<double> Logits(double[,] rows)
    {
        int n = rows.GetLength(0), v = rows.GetLength(1);
        var t = new Tensor<double>(new[] { n, v });
        for (int i = 0; i < n; i++)
            for (int j = 0; j < v; j++)
                t[i, j] = rows[i, j];
        return t;
    }

    [Fact]
    public void Perplexity_UniformOverV_EqualsV()
    {
        // Uniform logits over a 4-word vocab => the model is "torn between 4" => perplexity == 4.
        var logits = Logits(new double[,] { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } });
        double ppl = LanguageModelMetrics<double>.Perplexity(logits, new[] { 0, 3 });
        Assert.Equal(4.0, ppl, 6);
    }

    [Fact]
    public void Perplexity_ConfidentCorrect_ApproachesOne()
    {
        // A very peaked logit on the true token => near-zero surprise => perplexity ~ 1.
        var logits = Logits(new double[,] { { 20, 0, 0, 0 }, { 0, 0, 0, 20 } });
        double ppl = LanguageModelMetrics<double>.Perplexity(logits, new[] { 0, 3 });
        Assert.True(ppl < 1.01, $"expected perplexity ~1, got {ppl}");
        Assert.True(ppl >= 1.0 - 1e-9, $"perplexity must be >= 1, got {ppl}");
    }

    [Fact]
    public void Perplexity_ConfidentWrong_IsLarge()
    {
        // All confidence on the WRONG token => very high surprise => large perplexity.
        var logits = Logits(new double[,] { { 20, 0, 0, 0 } });
        double ppl = LanguageModelMetrics<double>.Perplexity(logits, new[] { 1 });
        Assert.True(ppl > 100.0, $"expected large perplexity for a confident wrong prediction, got {ppl}");
    }

    [Fact]
    public void CrossEntropy_MatchesLogOfPerplexity()
    {
        var logits = Logits(new double[,] { { 2.0, 1.0, 0.5, -1.0 }, { -0.5, 0.3, 1.7, 0.0 } });
        var targets = new[] { 0, 2 };
        double ce = LanguageModelMetrics<double>.CrossEntropy(logits, targets);
        double ppl = LanguageModelMetrics<double>.Perplexity(logits, targets);
        Assert.Equal(Math.Exp(ce), ppl, 6);
    }

    [Fact]
    public void Perplexity_FromProbabilities_Works()
    {
        // Rows already sum to 1; true-token probs are 0.5 and 0.25 => CE = mean(-ln p).
        var probs = Logits(new double[,] { { 0.5, 0.3, 0.2 }, { 0.25, 0.25, 0.5 } });
        double ppl = LanguageModelMetrics<double>.Perplexity(probs, new[] { 0, 1 }, fromLogits: false);
        double expected = Math.Exp((-Math.Log(0.5) - Math.Log(0.25)) / 2.0);
        Assert.Equal(expected, ppl, 6);
    }

    [Fact]
    public void TopKAccuracy_Top1_IsArgmaxAccuracy()
    {
        // Row 0 argmax=0 (target 0 => hit); row 1 argmax=1 but target 2 => miss. 1/2 = 0.5.
        var scores = Logits(new double[,] { { 3, 1, 0 }, { 0, 5, 2 } });
        double acc = LanguageModelMetrics<double>.TopKAccuracy(scores, new[] { 0, 2 }, k: 1);
        Assert.Equal(0.5, acc, 6);
    }

    [Fact]
    public void TopKAccuracy_Top2_IsMoreForgiving()
    {
        // Row 1: scores {0,5,2} => top-2 = {1,2}; target 2 is now a hit. Both rows hit => 1.0.
        var scores = Logits(new double[,] { { 3, 1, 0 }, { 0, 5, 2 } });
        double acc = LanguageModelMetrics<double>.TopKAccuracy(scores, new[] { 0, 2 }, k: 2);
        Assert.Equal(1.0, acc, 6);
    }

    [Fact]
    public void TopKAccuracy_HandlesTiesWithoutFalseHit()
    {
        // Tie between indices 0 and 1 (both 5); target is 2 (score 0). With k=1 the true token is
        // NOT in the single top slot regardless of tie-breaking => miss.
        var scores = Logits(new double[,] { { 5, 5, 0 } });
        double acc = LanguageModelMetrics<double>.TopKAccuracy(scores, new[] { 2 }, k: 1);
        Assert.Equal(0.0, acc, 6);
    }

    [Fact]
    public void InvalidArguments_Throw()
    {
        var logits = Logits(new double[,] { { 1, 2, 3 } });
        Assert.Throws<ArgumentException>(() => LanguageModelMetrics<double>.TopKAccuracy(logits, new[] { 0 }, k: 4));
        Assert.Throws<ArgumentException>(() => LanguageModelMetrics<double>.Perplexity(logits, new[] { 0, 1 })); // length mismatch
        Assert.Throws<ArgumentException>(() => LanguageModelMetrics<double>.Perplexity(logits, new[] { 9 }));    // target OOB

        // Null-argument paths (all three public methods guard predictions/targets).
        Assert.Throws<ArgumentNullException>(() => LanguageModelMetrics<double>.Perplexity(null!, new[] { 0 }));
        Assert.Throws<ArgumentNullException>(() => LanguageModelMetrics<double>.CrossEntropy(null!, new[] { 0 }));
        Assert.Throws<ArgumentNullException>(() => LanguageModelMetrics<double>.TopKAccuracy(logits, null!, k: 1));
    }
}
