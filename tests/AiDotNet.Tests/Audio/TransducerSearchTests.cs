using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Transducer decoding over the (t, u) lattice (Graves 2012, section 2.2) against hand-worked lattices. The joint
/// network is a table of Pr(k | t, labels so far); symbol 0 is the blank.
/// </summary>
public class TransducerSearchTests
{
    private static Tensor<double> Log(params double[] probabilities) =>
        new Tensor<double>(probabilities.Select(Math.Log).ToArray(), new[] { probabilities.Length });

    [Fact]
    public void Greedy_EmitsSeveralLabelsInOneFrame()
    {
        // One frame: after no labels "a" (0.8) wins, after "a" the label "b" (0.8) wins, after "ab" the blank (0.9)
        // ends the frame. A transducer emits "ab" in that single frame; per-frame argmax with CTC collapse (what the
        // model decoded before) can emit at most one label per frame.
        Tensor<double> LogProbs(IReadOnlyList<int> labels, int t) => labels.Count switch
        {
            0 => Log(0.1, 0.8, 0.1),
            1 => Log(0.1, 0.1, 0.8),
            _ => Log(0.9, 0.05, 0.05),
        };

        Assert.Equal(new List<int> { 1, 2 }, TransducerSearch.Greedy<double>(1, LogProbs, 0, maxSymbolsPerFrame: 10));
    }

    // With no labels the blank is the single most likely symbol at both frames; after any label the blank is 0.95.
    private static Tensor<double> MergingLattice(IReadOnlyList<int> labels, int t) =>
        labels.Count == 0 ? Log(0.4, 0.35, 0.25) : Log(0.95, 0.025, 0.025);

    [Fact]
    public void BeamSearch_SumsAlignments_WhereGreedyPicksTheBlank()
    {
        // Pr("") = 0.4 * 0.4 = 0.16. Pr("a") sums its two alignments: a at frame 0 (0.35 * 0.95 * 0.95 = 0.315875)
        // and a at frame 1 (0.4 * 0.35 * 0.95 = 0.133), 0.448875. Greedy takes the blank at both frames.
        Assert.Empty(TransducerSearch.Greedy<double>(2, MergingLattice, 0, maxSymbolsPerFrame: 2));

        var hypotheses = TransducerSearch.BeamSearch<double>(2, MergingLattice, 3, 0, beamWidth: 50, maxSymbolsPerFrame: 2);

        Assert.Equal(new[] { 1 }, hypotheses[0].Labels);
        Assert.Equal(Math.Log(0.448875), hypotheses[0].LogProbability, 9);

        var empty = hypotheses.Single(h => h.Labels.Length == 0);
        Assert.Equal(Math.Log(0.16), empty.LogProbability, 9);
    }

    [Fact]
    public void BeamSearch_RanksByLogProbabilityPerLabel()
    {
        // Graves 2012 returns the hypothesis with the highest log Pr(y) / |y|: "a" (log 0.448875) beats "b"
        // (log(0.25 * 0.95 * 0.95 + 0.4 * 0.25 * 0.95) = log 0.320625).
        var hypotheses = TransducerSearch.BeamSearch<double>(2, MergingLattice, 3, 0, beamWidth: 50, maxSymbolsPerFrame: 2);
        var b = hypotheses.Single(h => h.Labels.SequenceEqual(new[] { 2 }));

        Assert.Equal(Math.Log(0.320625), b.LogProbability, 9);
        Assert.True(hypotheses.IndexOf(b) > 0);
    }

    [Fact]
    public void BeamSearch_WithBeamOne_KeepsOneHypothesisPerFrame()
    {
        var hypotheses = TransducerSearch.BeamSearch<double>(2, MergingLattice, 3, 0, beamWidth: 1, maxSymbolsPerFrame: 2);
        Assert.Single(hypotheses);
    }
}
