// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Generation;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Generation;

public class AutoregressiveDecoderTests
{
    // Logits that always peak at `token`.
    private static Vector<float> PeakAt(int token, int vocab = 5)
    {
        var v = new float[vocab];
        for (int i = 0; i < vocab; i++) v[i] = 0.1f;
        v[token] = 9.0f;
        return new Vector<float>(v);
    }

    [Fact]
    public void Greedy_GeneratesMaxNewTokens_AndFeedsTokenBack()
    {
        var prevSeen = new List<int?>();
        var tokens = AutoregressiveDecoder<float>.Decode(
            stepLogits: prev => { prevSeen.Add(prev); return PeakAt(3); },
            maxNewTokens: 4,
            options: SamplingOptions.Greedy);

        Assert.Equal(new[] { 3, 3, 3, 3 }, tokens.ToArray());
        // First step gets null (prefill); subsequent steps get the previously-sampled token.
        Assert.Equal(new int?[] { null, 3, 3, 3 }, prevSeen.ToArray());
    }

    [Fact]
    public void StopsAtEndToken_WithoutEmittingIt()
    {
        const int eos = 4; // within the default vocab of 5
        int step = 0;
        var tokens = AutoregressiveDecoder<float>.Decode(
            stepLogits: _ => PeakAt(step++ == 2 ? eos : 1),
            maxNewTokens: 10,
            options: SamplingOptions.Greedy,
            isEndToken: t => t == eos);

        // steps 0,1 → token 1; step 2 → EOS → stop. EOS not emitted.
        Assert.Equal(new[] { 1, 1 }, tokens.ToArray());
    }

    [Fact]
    public void MaxNewTokensZero_ReturnsEmpty_AndNeverCallsStep()
    {
        int calls = 0;
        var tokens = AutoregressiveDecoder<float>.Decode(
            stepLogits: _ => { calls++; return PeakAt(0); },
            maxNewTokens: 0,
            options: SamplingOptions.Greedy);
        Assert.Empty(tokens);
        Assert.Equal(0, calls);
    }

    [Fact]
    public void Seeded_ProducesReproducibleSequence()
    {
        // A flat-ish distribution so sampling genuinely varies; seed must fix the whole sequence.
        Vector<float> dist() => new Vector<float>(new[] { 1.0f, 1.1f, 0.9f, 1.05f, 0.95f });
        IReadOnlyList<int> Run() => AutoregressiveDecoder<float>.Decode(
            stepLogits: _ => dist(), maxNewTokens: 12,
            options: new SamplingOptions { Temperature = 1.0, Seed = 2026 });

        Assert.Equal(Run().ToArray(), Run().ToArray());
    }

    [Fact]
    public void SuppressedToken_IsNotEmitted_ButConsumesStep()
    {
        // Step 0 peaks at the suppressed token 2 (greedy) → consumed, not emitted, prefix unchanged.
        // Steps 1..3 peak at token 1 → emitted. maxNewTokens=4 ⇒ exactly 3 emitted tokens.
        const int pad = 2;
        int step = 0;
        var prevSeen = new List<int?>();
        var tokens = AutoregressiveDecoder<float>.Decode(
            stepLogits: prev => { prevSeen.Add(prev); return PeakAt(step++ == 0 ? pad : 1); },
            maxNewTokens: 4,
            options: SamplingOptions.Greedy,
            suppressToken: t => t == pad);

        Assert.Equal(new[] { 1, 1, 1 }, tokens.ToArray());
        // The suppressed step did not advance `prev`, so step 1 still sees null (prefix unchanged).
        Assert.Equal(new int?[] { null, null, 1, 1 }, prevSeen.ToArray());
    }

    [Fact]
    public void NullStepLogits_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            AutoregressiveDecoder<float>.Decode(null!, 4));
    }
}
