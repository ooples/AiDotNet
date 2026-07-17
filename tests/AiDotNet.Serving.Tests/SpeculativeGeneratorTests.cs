using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Serving.Engine;
using AiDotNet.Serving.Engine.Speculative;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for greedy speculative decoding. The load-bearing invariant: speculative output is bit-identical to
/// plain greedy decoding of the same model, regardless of draft quality. Also verifies that speculation
/// actually accepts tokens (and beats one-token-per-pass) on predictable, repetitive output, plus the
/// prompt-lookup drafter's matching behavior.
/// </summary>
public class SpeculativeGeneratorTests
{
    private const int Vocab = 20;

    /// <summary>Strictly-increasing counter (no repeats ⇒ prompt-lookup finds nothing; tests correctness with
    /// ~zero acceptance).</summary>
    private sealed class CounterLm : ICausalLmModel<double>
    {
        public int VocabularySize => Vocab;
        public int? EosTokenId => null;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            var t = new Tensor<double>(new[] { 1, n, Vocab });
            for (int p = 0; p < n; p++)
            {
                int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, p]));
                t[0, p, (last + 1) % Vocab] = 1.0;
            }
            return t;
        }
    }

    /// <summary>Cyclic model (1→2→3→4→5→1…): its greedy output repeats with period 5, so prompt-lookup drafts
    /// correctly after one period ⇒ high acceptance.</summary>
    private sealed class CycleLm : ICausalLmModel<double>
    {
        public int VocabularySize => Vocab;
        public int? EosTokenId => null;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            var t = new Tensor<double>(new[] { 1, n, Vocab });
            for (int p = 0; p < n; p++)
            {
                int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, p]));
                t[0, p, (last % 5) + 1] = 1.0;
            }
            return t;
        }
    }

    // Reference plain-greedy decode (one token per forward pass).
    private static int[] PlainGreedy(ICausalLmModel<double> model, int[] prompt, int maxTokens)
    {
        var ctx = new List<int>(prompt);
        var gen = new List<int>();
        while (gen.Count < maxTokens)
        {
            var input = new Tensor<double>(new[] { 1, ctx.Count });
            for (int i = 0; i < ctx.Count; i++) input[0, i] = ctx[i];
            var logits = model.ForwardLogits(input);
            int last = ctx.Count - 1;
            int best = 0; double bestScore = logits[0, last, 0];
            for (int v = 1; v < Vocab; v++)
                if (logits[0, last, v] > bestScore) { bestScore = logits[0, last, v]; best = v; }
            gen.Add(best); ctx.Add(best);
        }
        return gen.ToArray();
    }

    private static SamplingParameters Greedy(int maxTokens, int minTokens = 0, IReadOnlyList<int>? stops = null)
        => new() { Temperature = 0.0, MaxTokens = maxTokens, MinTokens = minTokens, StopTokenIds = stops };

    // ---- Correctness invariant -----------------------------------------------------------

    [Theory]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(30)]
    public void SpeculativeGreedy_MatchesPlainGreedy_OnCyclicModel(int maxTokens)
    {
        var model = new CycleLm();
        var spec = new SpeculativeGenerator<double>(model, new PromptLookupDrafter(), maxDraftTokens: 4);

        var expected = PlainGreedy(model, new[] { 1 }, maxTokens);
        var actual = spec.Generate(new[] { 1 }, Greedy(maxTokens));

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void SpeculativeGreedy_MatchesPlainGreedy_OnNonRepetitiveModel()
    {
        var model = new CounterLm();
        var spec = new SpeculativeGenerator<double>(model, new PromptLookupDrafter(), maxDraftTokens: 4);

        var expected = PlainGreedy(model, new[] { 5 }, 12);
        var actual = spec.Generate(new[] { 5 }, Greedy(12), out var stats);

        Assert.Equal(expected, actual);
        // No repeats to draft from, but correctness still holds and every round still emits its bonus token.
        Assert.Equal(12, stats.GeneratedTokens);
    }

    // ---- Speculation actually pays off ---------------------------------------------------

    [Fact]
    public void CyclicModel_AcceptsDrafts_AndBeatsOnePassPerToken()
    {
        var spec = new SpeculativeGenerator<double>(new CycleLm(), new PromptLookupDrafter(), maxDraftTokens: 4);
        spec.Generate(new[] { 1 }, Greedy(40), out var stats);

        Assert.True(stats.AcceptedTokens > 0, "cyclic output should let the drafter accept tokens");
        Assert.True(stats.TargetForwardPasses < stats.GeneratedTokens,
            "fewer target passes than tokens generated proves the speedup");
        Assert.True(stats.TokensPerForwardPass > 1.0);
        Assert.InRange(stats.AcceptanceRate, 0.0, 1.0);
    }

    // ---- Stop conditions honored ---------------------------------------------------------

    [Fact]
    public void StopToken_EndsSpeculativeGeneration()
    {
        var spec = new SpeculativeGenerator<double>(new CycleLm(), new PromptLookupDrafter());
        // cycle from 1: 2,3,4,5,1,... stop at token 4.
        var gen = spec.Generate(new[] { 1 }, Greedy(100, stops: new[] { 4 }));
        Assert.Equal(new[] { 2, 3, 4 }, gen);
    }

    [Fact]
    public void MaxTokens_IsNeverExceeded()
    {
        var spec = new SpeculativeGenerator<double>(new CycleLm(), new PromptLookupDrafter(), maxDraftTokens: 8);
        var gen = spec.Generate(new[] { 1 }, Greedy(13));
        Assert.Equal(13, gen.Count);
    }

    [Fact]
    public void NonGreedySampling_Throws()
    {
        var spec = new SpeculativeGenerator<double>(new CycleLm());
        Assert.Throws<NotSupportedException>(() =>
            spec.Generate(new[] { 1 }, new SamplingParameters { Temperature = 1.0, MaxTokens = 5 }));
    }

    // ---- Prompt-lookup drafter -----------------------------------------------------------

    [Fact]
    public void PromptLookup_DraftsContinuationOfRepeatedNgram()
    {
        var drafter = new PromptLookupDrafter(maxNgram: 2, minNgram: 1);
        // Trailing "7 8" last occurred at index 1..2; it was followed there by 9,4,5 — so those are drafted
        // (draft accuracy is irrelevant; the target verifies every guess).
        var context = new[] { 1, 7, 8, 9, 4, 5, 7, 8 };
        var draft = drafter.Draft(context, maxDraftTokens: 3);
        Assert.Equal(new[] { 9, 4, 5 }, draft.ToArray());
    }

    [Fact]
    public void PromptLookup_NoMatch_ReturnsEmpty()
    {
        var drafter = new PromptLookupDrafter();
        var draft = drafter.Draft(new[] { 1, 2, 3, 4, 5 }, 3);
        Assert.Empty(draft);
    }

    [Fact]
    public void PromptLookup_RespectsMaxDraftTokens()
    {
        var drafter = new PromptLookupDrafter(maxNgram: 1, minNgram: 1);
        // repeated "9" pattern: 9 a b c 9 -> earlier "9" followed by a,b,c...
        var context = new[] { 9, 3, 4, 5, 6, 9 };
        var draft = drafter.Draft(context, maxDraftTokens: 2);
        Assert.Equal(2, draft.Count);
        Assert.Equal(new[] { 3, 4 }, draft.ToArray());
    }
}
