using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// CTC decoding helpers against references worked out by hand and by brute-force path enumeration
/// (Graves et al. 2006: p(l|x) is the sum over every frame alignment that collapses to l).
/// </summary>
public class SpeechRecognitionEngineOpsTests
{
    private static Tensor<double> LogitsFromProbabilities(double[,] probs)
    {
        int frames = probs.GetLength(0), vocab = probs.GetLength(1);
        var data = new double[frames * vocab];
        for (int t = 0; t < frames; t++)
            for (int v = 0; v < vocab; v++)
                data[t * vocab + v] = Math.Log(probs[t, v]);
        return new Tensor<double>(data, new[] { frames, vocab });
    }

    // Exact best label sequence by enumerating every alignment.
    private static List<int> BruteForceBestPrefix(double[,] probs, int blank)
    {
        int frames = probs.GetLength(0), vocab = probs.GetLength(1);
        var totals = new Dictionary<string, (double P, List<int> Labels)>();
        int count = (int)Math.Pow(vocab, frames);
        for (int code = 0; code < count; code++)
        {
            int rest = code;
            double p = 1;
            var labels = new List<int>();
            int prev = -1;
            for (int t = 0; t < frames; t++)
            {
                int s = rest % vocab;
                rest /= vocab;
                p *= probs[t, s];
                if (s != blank && s != prev) labels.Add(s);
                prev = s;
            }

            string key = string.Join(",", labels);
            totals[key] = totals.TryGetValue(key, out var old) ? (old.P + p, old.Labels) : (p, labels);
        }

        return totals.Values.OrderByDescending(v => v.P).First().Labels;
    }

    [Fact]
    public void PrefixBeamSearch_MergesAlignments_WhereGreedyAndUnmergedBeamsPickBlank()
    {
        // p("") = 0.16 but p("a") = 0.35*0.35 + 0.35*0.4 + 0.4*0.35 = 0.4025 (Hannun et al. 2014).
        var probs = new double[,] { { 0.4, 0.35, 0.25 }, { 0.4, 0.35, 0.25 } };
        var logits = LogitsFromProbabilities(probs);

        Assert.Empty(SpeechRecognitionEngineOps.CollapseCtcFrames(SpeechRecognitionEngineOps.FrameArgMax(logits), 0));
        Assert.Equal(new List<int> { 1 }, SpeechRecognitionEngineOps.CtcPrefixBeamSearch(logits, 2, 0));
        Assert.Equal(BruteForceBestPrefix(probs, 0), SpeechRecognitionEngineOps.CtcPrefixBeamSearch(logits, 2, 0));
    }

    [Fact]
    public void PrefixBeamSearch_RepeatAcrossBlankKeepsBothSymbols()
    {
        // a, blank, a -> "aa"; without the blank, a, a collapses to "a".
        var separated = new double[,] { { 0.05, 0.9, 0.05 }, { 0.9, 0.05, 0.05 }, { 0.05, 0.9, 0.05 } };
        var adjacent = new double[,] { { 0.05, 0.9, 0.05 }, { 0.05, 0.9, 0.05 } };

        Assert.Equal(new List<int> { 1, 1 }, SpeechRecognitionEngineOps.CtcPrefixBeamSearch(LogitsFromProbabilities(separated), 3, 0));
        Assert.Equal(new List<int> { 1 }, SpeechRecognitionEngineOps.CtcPrefixBeamSearch(LogitsFromProbabilities(adjacent), 3, 0));
    }

    [Fact]
    public void PrefixBeamSearch_MatchesBruteForce_OnThreeFrames()
    {
        var probs = new double[,] { { 0.3, 0.5, 0.2 }, { 0.45, 0.25, 0.3 }, { 0.2, 0.35, 0.45 } };
        Assert.Equal(BruteForceBestPrefix(probs, 0), SpeechRecognitionEngineOps.CtcPrefixBeamSearch(LogitsFromProbabilities(probs), 3, 0));
    }

    [Fact]
    public void FrameArgMax_SkipsNaNAndPrefersLowestIndexOnTies()
    {
        var logits = new Tensor<double>(new[] { double.NaN, 1.0, 1.0, 2.0, 2.0, 0.0 }, new[] { 2, 3 });
        Assert.Equal(new[] { 1, 0 }, SpeechRecognitionEngineOps.FrameArgMax(logits));
    }

    [Fact]
    public void ConcatFlat_JoinsChunksInFlatOrder()
    {
        var chunks = new List<Tensor<double>>
        {
            new Tensor<double>(new[] { 1.0, 2.0 }, new[] { 2 }),
            new Tensor<double>(new[] { 3.0, 4.0, 5.0, 6.0 }, new[] { 2, 2 }),
        };

        Assert.Equal(new[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, SpeechRecognitionEngineOps.ConcatFlat(chunks).ToArray());
    }
}
