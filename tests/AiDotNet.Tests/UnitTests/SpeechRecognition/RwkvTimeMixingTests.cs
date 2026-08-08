using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using AiDotNet.SpeechRecognition.ConformerFamily;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SpeechRecognition;

/// <summary>
/// Verifies the RWKV time-mixing recurrence behind <see cref="RWKVTransducer{T}"/>, for An and
/// Zhang, "Exploring RWKV for Memory Efficient and Low Latency Streaming ASR" (arXiv:2309.14758).
/// </summary>
/// <remarks>
/// The paper's whole claim is about streaming cost: full attention "is non-streamable and
/// computationally expensive, thus requiring modifications, such as chunking and caching". Two
/// properties decide whether an implementation actually delivers the alternative — the state must be
/// constant in the utterance length, and streaming must equal batch EXACTLY rather than approximate
/// it. Both are asserted here; a model that merely mentions RWKV would pass neither.
/// </remarks>
public class RwkvTimeMixingTests
{
    private const int Channels = 6;

    private static RwkvTimeMixing<double> Mixing(double decay = 0.5, double bonus = 1.0, double shift = 0.7)
        => new(Channels, decay, bonus, shift);

    private static Tensor<double> Sequence(int frames, int seed = 5)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([frames, Channels]);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    private static Vector<double> FrameOf(Tensor<double> seq, int t)
    {
        var v = new Vector<double>(Channels);
        for (int c = 0; c < Channels; c++) v[c] = seq[t * Channels + c];
        return v;
    }

    [Fact]
    public void StreamingEqualsBatch_Exactly()
    {
        // THE property. Frame-by-frame decoding is not a degraded mode of this encoder, it is the
        // same computation — which is what chunked attention can only approximate.
        var sequence = Sequence(frames: 12);

        var batch = Mixing().Forward(sequence);

        var streaming = Mixing();
        streaming.Reset();
        for (int t = 0; t < 12; t++)
        {
            var stepped = streaming.Step(FrameOf(sequence, t));
            for (int c = 0; c < Channels; c++)
            {
                Assert.Equal(batch[t * Channels + c], stepped[c], 12);
            }
        }
    }

    [Fact]
    public void StateSizeIsConstantInUtteranceLength()
    {
        // A transformer's KV cache grows with every frame; this is the cost the paper removes.
        var mixing = Mixing();
        int initial = mixing.StateSize;

        mixing.Reset();
        for (int t = 0; t < 500; t++)
        {
            mixing.Step(FrameOf(Sequence(frames: 1, seed: t), 0));
            Assert.Equal(initial, mixing.StateSize);
        }
    }

    [Fact]
    public void PastDecaysGeometrically_NotByTruncation()
    {
        // A larger decay must forget faster. Chunked attention instead truncates at a boundary; the
        // point of e^(-w) is that influence falls off smoothly.
        double InfluenceOfOpeningFrame(double decay)
        {
            var slow = new RwkvTimeMixing<double>(Channels, decay, 1.0, 1.0);
            slow.Reset();

            var loud = new Vector<double>(Channels);
            for (int c = 0; c < Channels; c++) loud[c] = 5.0;
            slow.Step(loud);

            var quiet = new Vector<double>(Channels);
            double last = 0.0;
            for (int t = 0; t < 6; t++) last = slow.Step(quiet)[0];
            return Math.Abs(last);
        }

        Assert.True(InfluenceOfOpeningFrame(2.0) < InfluenceOfOpeningFrame(0.05),
            "A larger time decay must leave less of the opening frame after the same number of steps.");
    }

    [Fact]
    public void ResetClearsTheUtterance()
    {
        // Two utterances through one instance must not contaminate each other.
        var sequence = Sequence(frames: 5, seed: 9);

        var mixing = Mixing();
        var first = mixing.Forward(sequence);
        var second = mixing.Forward(sequence);   // Forward resets

        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], second[i], 12);
    }

    [Fact]
    public void TokenShiftMixesTheCurrentFrameWithThePrevious()
    {
        // mu = 1 uses only the current frame; anything less blends in the previous one, so the two
        // must differ from the SECOND frame onward.
        var sequence = Sequence(frames: 4, seed: 3);
        var noShift = new RwkvTimeMixing<double>(Channels, 0.5, 1.0, 1.0).Forward(sequence);
        var shifted = new RwkvTimeMixing<double>(Channels, 0.5, 1.0, 0.5).Forward(sequence);

        // The first frame has no predecessor, so it must agree.
        for (int c = 0; c < Channels; c++) Assert.Equal(noShift[c], shifted[c], 12);

        bool differs = Enumerable.Range(Channels, noShift.Length - Channels)
            .Any(i => Math.Abs(noShift[i] - shifted[i]) > 1e-9);
        Assert.True(differs, "Token shift had no effect after the opening frame.");
    }

    [Fact]
    public void FirstFrameIsNotAttenuatedByAnAbsentPredecessor()
    {
        // Mixing frame one with a zero "previous" would silently damp the start of every utterance.
        // With no predecessor the shifted value is the frame itself, so mu cannot change frame one.
        var sequence = Sequence(frames: 3, seed: 17);
        var a = new RwkvTimeMixing<double>(Channels, 0.5, 1.0, 0.2).Forward(sequence);
        var b = new RwkvTimeMixing<double>(Channels, 0.5, 1.0, 0.9).Forward(sequence);

        for (int c = 0; c < Channels; c++) Assert.Equal(a[c], b[c], 12);
    }

    [Fact]
    public void OutputIsFiniteForLargeInputs()
    {
        // e^k overflows quickly; the recurrence clamps rather than producing infinities.
        var mixing = Mixing();
        var huge = new Vector<double>(Channels);
        for (int c = 0; c < Channels; c++) huge[c] = 1e6;

        mixing.Reset();
        for (int t = 0; t < 5; t++)
        {
            var output = mixing.Step(huge);
            for (int c = 0; c < Channels; c++)
            {
                Assert.False(double.IsNaN(output[c]) || double.IsInfinity(output[c]),
                    $"Channel {c} was non-finite at step {t}.");
            }
        }
    }

    [Fact]
    public void ConstructorRejectsParametersThatWouldDiverge()
    {
        // A negative decay makes e^(-w) > 1, amplifying the past every step until the state blows up
        // over a long utterance — precisely the regime streaming ASR runs in.
        Assert.Throws<ArgumentOutOfRangeException>(() => new RwkvTimeMixing<double>(Channels, -0.1, 1.0, 0.7));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RwkvTimeMixing<double>(0, 0.5, 1.0, 0.7));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RwkvTimeMixing<double>(Channels, 0.5, 1.0, 1.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RwkvTimeMixing<double>(Channels, 0.5, 1.0, -0.1));
    }
}
