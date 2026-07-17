using System;
using System.Collections.Generic;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Correctness tests for <see cref="LogitsSampler"/> — the decoding kernel that turns logits into a token id.
/// These pin the greedy path (argmax), the effect of each filter (top-k / top-p / min-p), the repetition /
/// presence / frequency penalties, and seeded-sampling reproducibility, against hand-computed expectations.
/// </summary>
public class LogitsSamplerTests
{
    private static readonly IReadOnlyList<int> NoContext = Array.Empty<int>();

    private static SamplingParameters Greedy() => new() { Temperature = 0.0 };

    private static Vector<double> V(params double[] logits) => new(logits);

    // ---- Greedy -------------------------------------------------------------------------

    [Fact]
    public void Greedy_ReturnsArgMax()
    {
        var logits = V(0.1, 3.5, 0.2, 1.0);
        int token = LogitsSampler.Sample(logits, Greedy(), NoContext, new Random(0));
        Assert.Equal(1, token);
    }

    [Fact]
    public void Greedy_IsDeterministic_RegardlessOfRng()
    {
        var logits = V(2.0, 5.0, 1.0);
        int a = LogitsSampler.Sample(logits, Greedy(), NoContext, new Random(1));
        int b = LogitsSampler.Sample(logits, Greedy(), NoContext, new Random(999));
        Assert.Equal(1, a);
        Assert.Equal(a, b);
    }

    // ---- Filters collapse to argmax -----------------------------------------------------

    [Fact]
    public void TopK1_AlwaysPicksMostProbable()
    {
        var logits = V(0.5, 4.0, 0.6, 3.9);
        var p = new SamplingParameters { Temperature = 1.0, TopK = 1 };
        for (int seed = 0; seed < 20; seed++)
            Assert.Equal(1, LogitsSampler.Sample(logits, p, NoContext, new Random(seed)));
    }

    [Fact]
    public void TinyTopP_CollapsesToArgMax()
    {
        // With one clearly dominant logit, a tiny nucleus keeps only that token.
        var logits = V(0.0, 10.0, 0.1, 0.2);
        var p = new SamplingParameters { Temperature = 1.0, TopP = 0.05 };
        for (int seed = 0; seed < 20; seed++)
            Assert.Equal(1, LogitsSampler.Sample(logits, p, NoContext, new Random(seed)));
    }

    [Fact]
    public void HighMinP_KeepsOnlyNearTopTokens()
    {
        // Two dominant tokens (index 1,3) far above the rest. A high min-p prunes the tail so only 1 and 3
        // can ever be sampled.
        var logits = V(-5.0, 5.0, -5.0, 5.0, -5.0);
        var p = new SamplingParameters { Temperature = 1.0, MinP = 0.5 };
        var seen = new HashSet<int>();
        for (int seed = 0; seed < 50; seed++)
            seen.Add(LogitsSampler.Sample(logits, p, NoContext, new Random(seed)));
        Assert.Subset(new HashSet<int> { 1, 3 }, seen);
    }

    // ---- Reproducibility ----------------------------------------------------------------

    [Fact]
    public void SeededSampling_IsReproducible()
    {
        var logits = V(1.0, 1.2, 0.9, 1.1, 0.8);
        var p = new SamplingParameters { Temperature = 1.5 };

        var first = new List<int>();
        var rng1 = new Random(12345);
        for (int i = 0; i < 30; i++) first.Add(LogitsSampler.Sample(logits, p, NoContext, rng1));

        var second = new List<int>();
        var rng2 = new Random(12345);
        for (int i = 0; i < 30; i++) second.Add(LogitsSampler.Sample(logits, p, NoContext, rng2));

        Assert.Equal(first, second);
    }

    // ---- Penalties ----------------------------------------------------------------------

    [Fact]
    public void RepetitionPenalty_CanFlipGreedyChoiceAwayFromRepeatedToken()
    {
        // Token 0 has the highest raw logit, but it already appears in the context. A strong repetition
        // penalty should divide its (positive) logit enough that token 1 wins the argmax.
        var logits = V(2.0, 1.8, 0.1);
        var context = new List<int> { 0, 0, 0 };

        int withoutPenalty = LogitsSampler.Sample(logits, Greedy(), context, new Random(0));
        Assert.Equal(0, withoutPenalty); // no penalty configured -> raw argmax

        var penalized = new SamplingParameters { Temperature = 0.0, RepetitionPenalty = 2.0 };
        int withPenalty = LogitsSampler.Sample(logits, penalized, context, new Random(0));
        Assert.Equal(1, withPenalty); // 2.0/2.0 = 1.0 < 1.8 -> token 1 now wins
    }

    [Fact]
    public void FrequencyPenalty_ScalesWithCount()
    {
        // Token 0 leads by a small margin but appears many times; frequency penalty scaled by count should
        // subtract enough to hand the argmax to token 1.
        var logits = V(1.5, 1.0, 0.0);
        var context = new List<int> { 0, 0, 0, 0, 0 }; // count = 5
        var p = new SamplingParameters { Temperature = 0.0, FrequencyPenalty = 0.2 };
        // 1.5 - 0.2*5 = 0.5 < 1.0 -> token 1
        int token = LogitsSampler.Sample(logits, p, context, new Random(0));
        Assert.Equal(1, token);
    }

    [Fact]
    public void PresencePenalty_AppliesOncePerSeenToken()
    {
        var logits = V(1.0, 0.7, 0.0);
        var context = new List<int> { 0, 0, 0 }; // appears, but presence applies once
        var p = new SamplingParameters { Temperature = 0.0, PresencePenalty = 0.5 };
        // 1.0 - 0.5 = 0.5 < 0.7 -> token 1
        int token = LogitsSampler.Sample(logits, p, context, new Random(0));
        Assert.Equal(1, token);
    }

    // ---- Distribution sanity ------------------------------------------------------------

    [Fact]
    public void UniformLogits_SampleRoughlyUniform()
    {
        var logits = V(0.0, 0.0, 0.0, 0.0);
        var p = new SamplingParameters { Temperature = 1.0 };
        var counts = new int[4];
        var rng = new Random(2024);
        const int draws = 20000;
        for (int i = 0; i < draws; i++) counts[LogitsSampler.Sample(logits, p, NoContext, rng)]++;

        double expected = draws / 4.0;
        foreach (int c in counts)
            Assert.InRange(c, expected * 0.85, expected * 1.15); // within 15% of uniform
    }

    [Fact]
    public void HigherLogit_SampledMoreOften()
    {
        var logits = V(0.0, 2.0); // softmax ~ [0.12, 0.88]
        var p = new SamplingParameters { Temperature = 1.0 };
        int hi = 0;
        var rng = new Random(7);
        const int draws = 10000;
        for (int i = 0; i < draws; i++) if (LogitsSampler.Sample(logits, p, NoContext, rng) == 1) hi++;
        Assert.InRange(hi / (double)draws, 0.82, 0.94); // near 0.88
    }

    // ---- Validation ---------------------------------------------------------------------

    [Fact]
    public void EmptyLogits_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            LogitsSampler.Sample(new Vector<double>(Array.Empty<double>()), Greedy(), NoContext, new Random(0)));
    }
}
