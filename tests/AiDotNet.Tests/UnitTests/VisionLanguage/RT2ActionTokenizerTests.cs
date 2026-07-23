using System;
using System.Threading.Tasks;
using AiDotNet.VisionLanguage.Robotics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Unit tests for RT-2's 256-bin action tokenizer (paper §3.2 — Brohan et al. 2023, arXiv:2307.15818).
/// These cover the architectural-correctness pieces that DO NOT depend on trained model weights:
/// bin/token mapping, round-trip precision, clamping behaviour, and greedy logit selection.
/// </summary>
public class RT2ActionTokenizerTests
{
    [Fact(Timeout = 30000)]
    public async Task TokenIdOffset_PlacesBinsAtTopOfVocabulary()
    {
        await Task.Yield();
        var tokenizer = new RT2ActionTokenizer<double>(actionDim: 8, numBins: 256, vocabSize: 32000);

        Assert.Equal(32000 - 256, tokenizer.TokenIdOffset);
        Assert.Equal(32000, tokenizer.TokenIdEndExclusive);
    }

    [Fact(Timeout = 30000)]
    public async Task EncodeDecode_RoundTrip_WithinBinPrecision()
    {
        await Task.Yield();
        var tokenizer = new RT2ActionTokenizer<double>(actionDim: 3, numBins: 256);

        var original = new Tensor<double>([3]);
        original[0] = 0.5;
        original[1] = -0.25;
        original[2] = 0.875;

        var tokens = tokenizer.EncodeAction(original);
        var decoded = tokenizer.DecodeAction(tokens);

        // Each bin spans 2.0 / 256 = 0.0078; decoded should land at bin centre, within half a bin.
        double binWidth = 2.0 / 256;
        for (int d = 0; d < 3; d++)
        {
            double diff = Math.Abs(original[d] - decoded[d]);
            Assert.True(diff <= binWidth, $"Dim {d}: original={original[d]}, decoded={decoded[d]}, diff={diff} exceeds half-bin {binWidth}.");
        }
    }

    [Fact(Timeout = 30000)]
    public async Task EncodeAction_ClampsAboveMaxToLastBin()
    {
        await Task.Yield();
        var tokenizer = new RT2ActionTokenizer<double>(actionDim: 2, numBins: 256);

        var outOfRange = new Tensor<double>([2]);
        outOfRange[0] = 5.0;
        outOfRange[1] = -5.0;

        var tokens = tokenizer.EncodeAction(outOfRange);

        Assert.Equal(tokenizer.TokenIdOffset + 255, tokens[0]);
        Assert.Equal(tokenizer.TokenIdOffset, tokens[1]);
    }

    [Fact(Timeout = 30000)]
    public async Task GreedyActionToken_PicksHighestLogitInWindow()
    {
        await Task.Yield();
        var tokenizer = new RT2ActionTokenizer<double>(actionDim: 8, numBins: 256, vocabSize: 32000);

        var logits = new Tensor<double>([32000]);
        // Plant a clear winner at bin index 100 (token id = 32000 - 256 + 100 = 31844).
        // Other action-bin slots are small; non-bin slots are even larger but should be ignored.
        for (int i = 0; i < 32000; i++) logits[i] = 100.0; // saturate non-bin region
        for (int i = tokenizer.TokenIdOffset; i < tokenizer.TokenIdEndExclusive; i++)
            logits[i] = 0.0;
        logits[tokenizer.TokenIdOffset + 100] = 50.0;

        int picked = tokenizer.GreedyActionToken(logits);

        Assert.Equal(tokenizer.TokenIdOffset + 100, picked);
    }

    [Fact(Timeout = 30000)]
    public async Task EncodeHorizon_ProducesActionDimTimesHorizonTokens()
    {
        await Task.Yield();
        var tokenizer = new RT2ActionTokenizer<double>(actionDim: 8, numBins: 256);
        var horizonAction = new Tensor<double>([8 * 4]);
        for (int i = 0; i < 32; i++) horizonAction[i] = (i % 8) * 0.1 - 0.4;

        var tokens = tokenizer.EncodeHorizon(horizonAction, horizon: 4);

        Assert.Equal(32, tokens.Length);
        for (int t = 0; t < tokens.Length; t++)
            Assert.InRange(tokens[t], tokenizer.TokenIdOffset, tokenizer.TokenIdEndExclusive - 1);
    }

    [Fact(Timeout = 30000)]
    public async Task DecodeAction_OutOfWindowToken_FallsBackToMidpoint()
    {
        await Task.Yield();
        var tokenizer = new RT2ActionTokenizer<double>(actionDim: 1, numBins: 256);

        var malformedTokens = new int[] { 0 };
        var decoded = tokenizer.DecodeAction(malformedTokens);

        // Midpoint of [-1, 1] is approximately 0 (256/2 = bin 128 → centre at 0.5/256 above midline).
        Assert.InRange(decoded[0], -binCentreSlack, binCentreSlack);
    }

    [Fact(Timeout = 30000)]
    public async Task Constructor_RejectsVocabSmallerThanNumBins()
    {
        await Task.Yield();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new RT2ActionTokenizer<double>(actionDim: 8, numBins: 256, vocabSize: 100));
    }

    [Fact(Timeout = 30000)]
    public async Task PerDimensionRange_HonoursAsymmetricBounds()
    {
        await Task.Yield();
        var minRange = new[] { 0.0, -10.0 };
        var maxRange = new[] { 1.0, 10.0 };
        var tokenizer = new RT2ActionTokenizer<double>(actionDim: 2, numBins: 256, minRange: minRange, maxRange: maxRange);

        var action = new Tensor<double>([2]);
        action[0] = 0.5;
        action[1] = 0.0;

        var tokens = tokenizer.EncodeAction(action);
        var decoded = tokenizer.DecodeAction(tokens);

        // Dim 0: 0.5 in [0, 1] → bin ~128; decode → ~0.5
        Assert.InRange(decoded[0], 0.4, 0.6);
        // Dim 1: 0.0 in [-10, 10] → bin ~128; decode → ~0.0
        Assert.InRange(decoded[1], -1.0, 1.0);
    }

    private const double binCentreSlack = 0.01;
}
