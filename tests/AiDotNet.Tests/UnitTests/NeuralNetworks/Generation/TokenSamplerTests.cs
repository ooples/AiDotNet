// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Generation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Generation;

public class TokenSamplerTests
{
    private static Vector<float> V(params float[] xs) => new Vector<float>(xs);

    [Fact]
    public void ArgMax_ReturnsIndexOfMaxLogit()
    {
        Assert.Equal(2, TokenSampler<float>.ArgMax(V(0.1f, 0.5f, 9.0f, -3f)));
    }

    [Fact]
    public void Greedy_IsDeterministicArgMax()
    {
        var logits = V(1f, 7f, 2f, 3f);
        for (int i = 0; i < 20; i++)
            Assert.Equal(1, TokenSampler<float>.Sample(logits, SamplingOptions.Greedy));
    }

    [Fact]
    public void TopK1_AlwaysPicksArgMax_RegardlessOfSeed()
    {
        var logits = V(0.2f, 0.2f, 5.0f, 0.2f, 0.2f); // index 2 dominates
        for (int seed = 0; seed < 25; seed++)
        {
            var opts = new SamplingOptions { Temperature = 1.5, TopK = 1, Seed = seed };
            Assert.Equal(2, TokenSampler<float>.Sample(logits, opts));
        }
    }

    [Fact]
    public void TopP_Tiny_CollapsesToArgMax()
    {
        var logits = V(0.1f, 8.0f, 0.2f, 0.3f); // index 1 has nearly all mass
        for (int seed = 0; seed < 25; seed++)
        {
            var opts = new SamplingOptions { Temperature = 1.0, TopP = 0.001, Seed = seed };
            Assert.Equal(1, TokenSampler<float>.Sample(logits, opts));
        }
    }

    [Fact]
    public void Seeded_IsReproducible()
    {
        var logits = V(1f, 1.2f, 0.8f, 1.1f, 0.9f);
        var a = new SamplingOptions { Temperature = 1.0, Seed = 1234 };
        int first = TokenSampler<float>.Sample(logits, a);
        // Same seed → fresh seeded RNG → identical first draw → identical token.
        Assert.Equal(first, TokenSampler<float>.Sample(logits, new SamplingOptions { Temperature = 1.0, Seed = 1234 }));
    }

    [Fact]
    public void Sample_AlwaysReturnsValidIndex()
    {
        var logits = V(0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f);
        var rng = new Random(7);
        var opts = new SamplingOptions { Temperature = 1.0 };
        for (int i = 0; i < 200; i++)
        {
            int t = TokenSampler<float>.Sample(logits, opts, rng);
            Assert.InRange(t, 0, logits.Length - 1);
        }
    }

    [Fact]
    public void Temperature_ChangesDistributionSpread()
    {
        // Low temperature should concentrate on the argmax far more than high temperature.
        var logits = V(0f, 2f, 0f, 0f); // index 1 favored
        var rngLow = new Random(99);
        var rngHigh = new Random(99);
        int low = 0, high = 0;
        for (int i = 0; i < 400; i++)
        {
            if (TokenSampler<float>.Sample(logits, new SamplingOptions { Temperature = 0.2 }, rngLow) == 1) low++;
            if (TokenSampler<float>.Sample(logits, new SamplingOptions { Temperature = 3.0 }, rngHigh) == 1) high++;
        }
        Assert.True(low > high, $"low-temp argmax rate {low} should exceed high-temp {high}");
    }
}
