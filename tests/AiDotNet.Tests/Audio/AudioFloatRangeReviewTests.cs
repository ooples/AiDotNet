using System;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Speaker;
using AiDotNet.Audio.VoiceActivity;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Scores and statistics whose squares overflow the element type (float: above 3.4e38) when they are reduced
/// in T. Each expected value is the double reference, which these inputs do not overflow.
/// </summary>
public class AudioFloatRangeReviewTests
{
    private static SpeakerEmbedding<float> Embedding(params float[] values) => new SpeakerEmbedding<float> { Vector = values };

    [Fact]
    public void SpeakerEmbedding_CosineAndEuclid_MatchTheDoubleReferenceForLargeMagnitudes()
    {
        var a = Embedding(1e20f, 2e20f, 3e20f);
        var b = Embedding(2e20f, 4e20f, 6e20f);

        Assert.Equal(1.0, a.CosineSimilarity(b), 5);
        // ||a - b|| = 1e20 * sqrt(14)
        Assert.Equal(1.0, a.EuclideanDistance(b) / (1e20 * Math.Sqrt(14.0)), 5);
    }

    [Fact]
    public void SpeakerVerifier_VerifyAndIdentify_ScoreParallelEmbeddingsAsOne()
    {
        var verifier = new SpeakerVerifier<float>();
        verifier.Enroll("alice", Embedding(1e20f, 2e20f, 3e20f));
        verifier.Enroll("bob", Embedding(3e20f, -1e20f, 0f));
        var probe = Embedding(2e20f, 4e20f, 6e20f);

        Assert.Equal(1.0, verifier.Verify("alice", probe).Score, 4);

        var identified = verifier.Identify(probe);
        Assert.Equal("alice", identified.IdentifiedSpeakerId);
        Assert.Equal(1.0, identified.TopScore, 4);
        // cos(bob, probe) = (3 - 2) / (sqrt(10) sqrt(14))
        double bob = 1.0 / (Math.Sqrt(10.0) * Math.Sqrt(14.0));
        Assert.Contains(identified.Matches, m => m.SpeakerId == "bob" && Math.Abs(m.Score - bob) < 1e-4);
    }

    [Fact]
    public void EnergyBasedVad_Energy_IsTheMeanSquareOfLargeSamples()
    {
        var vad = new EnergyBasedVad<float>();

        double energy = vad.ComputeEnergy(new[] { 1e20f, -1e20f, 2e20f, 0f });

        // (1 + 1 + 4 + 0) / 4 * 1e40
        Assert.Equal(1.0, energy / 1.5e40, 5);
    }

    [Fact]
    public void EnergyBasedVad_SpectralFlatness_IsScaleInvariantUpToLargeAmplitudes()
    {
        var unit = new double[256];
        var scaled = new float[256];
        for (int i = 0; i < unit.Length; i++)
        {
            unit[i] = Math.Sin(1.3 * i) + 0.5 * Math.Cos(0.37 * i) + ((i * 7919 % 13) - 6) / 6.0;
            scaled[i] = (float)(unit[i] * 1e18);
        }

        double reference = new EnergyBasedVad<double>().ComputeSpectralFlatness(unit);
        double large = new EnergyBasedVad<float>().ComputeSpectralFlatness(scaled);

        Assert.InRange(reference, 0.01, 1.0);
        Assert.Equal(1.0, large / reference, 3);
    }

    [Fact]
    public void ConformerNormalizePerBin_StandardizesLargeColumns()
    {
        // One bin over two frames, [-1.5e19, 1.5e19]: mean 0, population deviation 1.5e19, so [-1, 1].
        // Its squares (2.25e38) sum past float's 3.4e38.
        var mel = new Tensor<float>(new[] { -1.5e19f, 1.5e19f }, new[] { 2, 1 });

        var normalized = ConformerAsrNetwork<float>.NormalizePerBin(AiDotNetEngine.Current, mel);

        Assert.Equal(-1.0, normalized[0, 0], 5);
        Assert.Equal(1.0, normalized[1, 0], 5);
    }
}
