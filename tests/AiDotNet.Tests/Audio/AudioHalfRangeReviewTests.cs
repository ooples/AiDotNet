using System;
using AiDotNet.Audio.Speaker;
using AiDotNet.Audio.SpeechRecognition;
using AiDotNet.Audio.VoiceActivity;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Half-precision ranges of the speaker scores, the VAD threshold and the batch-mean transducer loss. Half's largest
/// finite value is 65504, so a norm, a sum or a rounded threshold formed in Half can leave the range (or move) where
/// the double reference does not. Each expected value is that double reference.
/// </summary>
public class AudioHalfRangeReviewTests
{
    private static readonly INumericOperations<Half> HalfOps = MathHelper.GetNumericOperations<Half>();

    private static Half H(double value) => HalfOps.FromDouble(value);

    private static double D(Half value) => HalfOps.ToDouble(value);

    // A few Half roundings (2^-11 each) of values at most 1.
    private const double HalfTolerance = 2e-3;

    private static void AssertNear(double expected, double actual) =>
        Assert.True(Math.Abs(actual - expected) <= HalfTolerance, $"Expected {expected}, got {actual}.");

    private static SpeakerEmbedding<Half> Embedding(params double[] values) =>
        new SpeakerEmbedding<Half> { Vector = Array.ConvertAll(values, H) };

    [Fact]
    public void SpeakerEmbedding_CosineSimilarity_InHalf_ScoresParallelVectorsAsOne()
    {
        // ||a|| = 4e4 sqrt(3) = 69282 is past Half's 65504: a norm returned in Half is infinite, and dividing by it
        // scored these parallel vectors 0 instead of 1.
        var a = Embedding(4e4, 4e4, 4e4);
        var b = Embedding(2e4, 2e4, 2e4);

        AssertNear(1.0, a.CosineSimilarity(b));
    }

    [Fact]
    public void SpeakerEmbedding_EuclideanDistance_InHalf_IsTheDoubleDistance()
    {
        // The differences are 6e4 each, so the distance is 6e4 sqrt(3) = 103923: infinite when returned in Half.
        var a = Embedding(4e4, 4e4, 4e4);
        var b = Embedding(-2e4, -2e4, -2e4);

        Assert.Equal(1.0, a.EuclideanDistance(b) / (6e4 * Math.Sqrt(3.0)), 6);
    }

    [Fact]
    public void SpeakerScoring_L2Normalize_InHalf_GivesAUnitVector()
    {
        // NormalizeEmbedding divided by a Half norm of infinity and returned zeros, so an enrolled profile with these
        // entries scored 0 against everything.
        var normalized = SpeakerScoring.L2Normalize(AiDotNetEngine.Current, new Tensor<Half>(new[] { H(4e4), H(4e4), H(4e4) }, new[] { 3 }));

        double expected = 1.0 / Math.Sqrt(3.0);
        Assert.Equal(new[] { 3 }, normalized.Shape.ToArray());
        for (int i = 0; i < 3; i++)
        {
            AssertNear(expected, D(normalized[i]));
        }
    }

    [Fact]
    public void SpeakerScoring_Cosine_InHalf_MatchesTheDoubleReference()
    {
        // cos([4e4, 0, 4e4], [4e4, 4e4, 0]) = 1/2
        var a = new Tensor<Half>(new[] { H(4e4), H(0), H(4e4) }, new[] { 3 });
        var b = new Tensor<Half>(new[] { H(4e4), H(4e4), H(0) }, new[] { 3 });

        Half cosine = SpeakerScoring.Cosine(AiDotNetEngine.Current, a, b, out double normProduct);

        AssertNear(0.5, D(cosine));
        AssertNear(1.0, normProduct / (2 * 16e8));
    }

    [Fact]
    public void VadSpeechMask_ComparesHalfProbabilitiesAgainstTheExactThreshold()
    {
        // Half holds 0.2 as 0.199951171875, below the threshold 0.2; comparing against the threshold rounded to Half
        // (also 0.199951171875) counted that frame as speech. (0.3 would not do: Half rounds it up, to 0.300048828125.)
        Assert.Equal(0.199951171875, D(H(0.2)));
        var mask = VadEngineOps.SpeechMask(new[] { H(0.2), H(0.5), H(0.1) }, 0.2);

        Assert.Equal(new[] { 0, 1, 0 }, mask);
    }

    [Fact]
    public void MeanLoss_InHalf_AveragesLossesWhoseSumOverflows()
    {
        // Two utterances of loss 60000 each: their sum (120000) is past Half's 65504, their mean is not.
        var mean = SpeechRecognitionEngineOps.MeanLoss(AiDotNetEngine.Current, 2, _ => new Tensor<Half>(new[] { H(60000) }, new[] { 1 }));

        Assert.Equal(60000.0, D(mean.GetFlat(0)), 6);
    }
}
