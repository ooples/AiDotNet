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

    // 4096 entries around 100 (and a second vector correlated 0.6 with the first), as Half stores them.
    private static (Half[] A, Half[] B) LongHalfVectors()
    {
        const int dim = 4096;
        var random = new Random(7);
        double Normal() => Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble());
        var a = new Half[dim];
        var b = new Half[dim];
        for (int i = 0; i < dim; i++)
        {
            double x = 100.0 * Normal();
            a[i] = H(x);
            b[i] = H(0.6 * x + 80.0 * Normal());
        }

        return (a, b);
    }

    private static double DoubleCosine(Half[] a, Half[] b)
    {
        double dot = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double x = D(a[i]), y = D(b[i]);
            dot += x * y;
            aa += x * x;
            bb += y * y;
        }

        return dot / Math.Sqrt(aa * bb);
    }

    [Fact]
    public void SpeakerEmbedding_CosineSimilarity_InHalf_At4096Entries_StaysInRangeAndMatchesTheDoubleReference()
    {
        // Reduced in Half, the rounding of 4096 additions put the cosine of this vector with itself at 1.0009765625
        // (outside [-1, 1]: its arc cosine is NaN) and the cross score 1e-4 off. The method returns a double and
        // master computed it in double.
        var (a, b) = LongHalfVectors();
        var first = new SpeakerEmbedding<Half> { Vector = a };
        var second = new SpeakerEmbedding<Half> { Vector = b };
        var opposite = new SpeakerEmbedding<Half> { Vector = Array.ConvertAll(a, v => H(-D(v))) };

        double self = first.CosineSimilarity(first);
        Assert.InRange(self, -1.0, 1.0);
        Assert.Equal(1.0, self, 12);

        double negative = first.CosineSimilarity(opposite);
        Assert.InRange(negative, -1.0, 1.0);
        Assert.Equal(-1.0, negative, 12);

        double cross = first.CosineSimilarity(second);
        Assert.InRange(cross, -1.0, 1.0);
        Assert.Equal(DoubleCosine(a, b), cross, 12);
    }

    [Fact]
    public void SpeakerScoring_Cosine_InHalf_At4096Entries_IsTheDoubleReferenceRoundedOnce()
    {
        // The T-valued score the speaker models compare (ComputeCosineSimilarity): the double cosine rounded to Half.
        var (a, b) = LongHalfVectors();
        var ta = new Tensor<Half>(a, new[] { a.Length });
        var tb = new Tensor<Half>(b, new[] { b.Length });

        double self = D(SpeakerScoring.Cosine(AiDotNetEngine.Current, ta, ta, out _));
        Assert.Equal(1.0, self);

        double cross = D(SpeakerScoring.Cosine(AiDotNetEngine.Current, ta, tb, out _));
        Assert.Equal(D(H(DoubleCosine(a, b))), cross);
    }

    [Fact]
    public void SpeakerScoring_Cosine_OnAViewInsideItsBuffer_ScoresTheView()
    {
        // [decoy | vector] narrowed to the vector: the score is that of the vector, not of the buffer's first entries.
        var engine = AiDotNetEngine.Current;
        var vector = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        var view = engine.TensorNarrow(new Tensor<float>(new[] { -4f, 3f, -2f, 1f, 1f, 2f, 3f, 4f }, new[] { 8 }), 0, 4, 4);

        Assert.Equal(1.0, (double)SpeakerScoring.Cosine(engine, view, vector, out _), 6);
        Assert.Equal(1.0, SpeakerScoring.CosineInDouble(engine, vector, view, out _), 12);
    }

    [Fact]
    public void SpeakerScoring_L2Normalize_InHalf_At4096Entries_HasUnitLengthAndKeepsItsShape()
    {
        // With the norm summed in Half the normalized vector had length 1.0005; from the double norm each entry is
        // the exact quotient rounded to Half once, and the length is 1 to a few parts in 1e5.
        var (a, _) = LongHalfVectors();
        var normalized = SpeakerScoring.L2Normalize(AiDotNetEngine.Current, new Tensor<Half>(a, new[] { 1, a.Length }));

        Assert.Equal(new[] { 1, a.Length }, normalized.Shape.ToArray());
        double norm = 0;
        foreach (Half v in a) norm += D(v) * D(v);
        norm = Math.Sqrt(norm);
        var values = normalized.ToArray();
        double squares = 0;
        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal(D(H(D(a[i]) / norm)), D(values[i]));
            squares += D(values[i]) * D(values[i]);
        }

        Assert.True(Math.Abs(Math.Sqrt(squares) - 1.0) <= 1e-4, $"length {Math.Sqrt(squares)}");
    }

    [Fact]
    public void SpeakerScoring_InFloatAndDouble_KeepsTheScoresToRounding()
    {
        // cos([1, 2, 3], [3, 2, 1]) = 10 / 14, and a vector with itself is exactly inside the range.
        var engine = AiDotNetEngine.Current;
        var fa = new Tensor<float>(new[] { 1f, 2f, 3f }, new[] { 3 });
        var fb = new Tensor<float>(new[] { 3f, 2f, 1f }, new[] { 3 });
        var da = new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 });
        var db = new Tensor<double>(new[] { 3.0, 2.0, 1.0 }, new[] { 3 });

        Assert.Equal((float)(10.0 / 14.0), SpeakerScoring.Cosine(engine, fa, fb, out double floatNorms));
        Assert.Equal(14.0, floatNorms, 12);
        Assert.Equal(10.0 / 14.0, SpeakerScoring.Cosine(engine, da, db, out double doubleNorms), 15);
        Assert.Equal(14.0, doubleNorms, 12);
        Assert.InRange(SpeakerScoring.Cosine(engine, da, da, out _), -1.0, 1.0);
        Assert.Equal(0f, SpeakerScoring.Cosine(engine, fa, new Tensor<float>(new[] { 3 }), out _));
        Assert.Equal(0.0, new SpeakerEmbedding<float> { Vector = Array.Empty<float>() }.CosineSimilarity(new SpeakerEmbedding<float> { Vector = Array.Empty<float>() }));

        var unit = SpeakerScoring.L2Normalize(engine, da).ToArray();
        double length = Math.Sqrt(14.0);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal((i + 1) / length, unit[i], 15);
        }

        Assert.Equal(new[] { 0f, 0f }, SpeakerScoring.L2Normalize(engine, new Tensor<float>(new[] { 2 })).ToArray());
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
