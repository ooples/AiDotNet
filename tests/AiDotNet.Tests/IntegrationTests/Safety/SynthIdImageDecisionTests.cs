using System;
using System.Linq;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// The SynthID-style watermarker's calibrated decision, checked on an untrained network so that no training is
/// needed. The conformal p-values and the three-state verdict are compared against a brute-force count over the
/// calibration scores. The calibration must also survive WithKey and serialization.
/// </summary>
public class SynthIdImageDecisionTests
{
    private const int Size = 32;

    private static SynthIdImageWatermarker<float> Untrained(double alpha = 0.1) =>
        new(new SynthIdImageOptions { ImageSize = Size, SecretBits = 16, Seed = 3, FalsePositiveRate = alpha }, "decision-key");

    private static Tensor<float> Images(int seed, int count) =>
        StegaStampWatermarkerTests.Images(RandomHelper.CreateSeededRandom(seed), count);

    private const double LogitTolerance = 1e-6;

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task Decide_MatchesBruteForceConformalPValues_AndTheThreeStateRule()
    {
        await System.Threading.Tasks.Task.Yield();
        var model = Untrained();
        // An untrained detection logit is an arbitrary function of the image. The p-values only rank it against the
        // calibration scores, so two disjoint image sets are enough to check the arithmetic.
        var clean = Images(1, 24);
        var marked = Images(2, 24);
        model.Calibrate(clean, marked);
        double[] cleanScores = model.DetectionLogits(clean), markedScores = model.DetectionLogits(marked);

        var probes = Images(3, 12);
        var probeScores = model.DetectionLogits(probes);
        for (int i = 0; i < probes.Shape[0]; i++)
        {
            var d = model.Decide(probes.Slice(i));
            // The batch and the single image round differently in float (1.5e-8 apart on CI); decimal-place
            // rounding can split such a pair across a boundary, so compare within an absolute tolerance.
            Assert.Equal(probeScores[i], d.DetectionLogit, LogitTolerance);
            double cleanP = (1.0 + cleanScores.Count(c => c >= d.DetectionLogit)) / (cleanScores.Length + 1.0);
            double markedP = (1.0 + markedScores.Count(m => m <= d.DetectionLogit)) / (markedScores.Length + 1.0);
            Assert.Equal(cleanP, d.CleanPValue, 12);
            Assert.Equal(markedP, d.WatermarkedPValue, 12);
            var expected = cleanP <= 0.1 && markedP > 0.1 ? WatermarkVerdict.Watermarked
                : markedP <= 0.1 && cleanP > 0.1 ? WatermarkVerdict.NotWatermarked
                : WatermarkVerdict.Abstain;
            Assert.Equal(expected, d.Verdict);
        }
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task CleanCalibrationImages_AreCalledWatermarked_AtMostAlphaOfTheTime()
    {
        await System.Threading.Tasks.Task.Yield();
        // Scored one at a time, the clean calibration images are exchangeable with themselves, so at most
        // floor(alpha (n + 1)) of them can have a clean p-value at or below alpha. A calibration scored through a
        // batched forward broke this: rounding put every single-image score above the batch's scores.
        const int n = 39;
        var model = Untrained(alpha: 0.1);
        var clean = Images(11, n);
        model.Calibrate(clean, Images(12, n));
        int flagged = Enumerable.Range(0, n).Count(i => model.Decide(clean.Slice(i)).CleanPValue <= 0.1);
        Assert.True(flagged <= (int)Math.Floor(0.1 * (n + 1)), $"{flagged} of {n} clean calibration images have p <= 0.1.");
    }

    [Fact(Timeout = 60000)]
    public async System.Threading.Tasks.Task Decide_BeforeCalibration_Throws()
    {
        await System.Threading.Tasks.Task.Yield();
        var model = Untrained();
        Assert.False(model.IsCalibrated);
        Assert.Throws<InvalidOperationException>(() => model.Decide(Images(4, 1).Slice(0)));
    }

    [Fact(Timeout = 60000)]
    public async System.Threading.Tasks.Task TooFewCalibrationImages_ForTheLevel_Abstain()
    {
        await System.Threading.Tasks.Task.Yield();
        // With n calibration images the smallest p-value is 1 / (n + 1). At alpha = 0.01 and n = 24 no hypothesis can be
        // rejected, so every decision must abstain rather than claim a level it cannot reach.
        var model = Untrained(alpha: 0.01);
        model.Calibrate(Images(5, 24), Images(6, 24));
        var probes = Images(7, 8);
        for (int i = 0; i < probes.Shape[0]; i++)
            Assert.Equal(WatermarkVerdict.Abstain, model.Decide(probes.Slice(i)).Verdict);
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task Calibration_SurvivesWithKey_AndSerialization()
    {
        await System.Threading.Tasks.Task.Yield();
        var model = Untrained();
        model.Calibrate(Images(8, 16), Images(9, 16));
        var probe = Images(10, 1).Slice(0);
        var expected = model.Decide(probe);

        var keyed = model.WithKey("another-key");
        Assert.True(keyed.IsCalibrated);
        Assert.Equal(expected.CleanPValue, keyed.Decide(probe).CleanPValue, 12);

        var restored = Untrained();
        restored.Deserialize(model.Serialize());
        Assert.True(restored.IsCalibrated);
        var again = restored.Decide(probe);
        Assert.Equal(expected.DetectionLogit, again.DetectionLogit, LogitTolerance);
        Assert.Equal(expected.CleanPValue, again.CleanPValue, 12);
        Assert.Equal(expected.WatermarkedPValue, again.WatermarkedPValue, 12);
    }

    [Fact(Timeout = 60000)]
    public async System.Threading.Tasks.Task Options_AreValidated_AndTheDefaultsAreSynthIdOsSizes()
    {
        await System.Threading.Tasks.Task.Yield();
        var defaults = new SynthIdImageOptions();
        Assert.Equal(512, defaults.ImageSize);
        Assert.Equal(136, defaults.SecretBits);
        Assert.False(defaults.UseErrorCorrection);
        Assert.Equal(0.001, defaults.FalsePositiveRate);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SynthIdImageWatermarker<float>(new SynthIdImageOptions { ImageSize = Size, SecretBits = 16, FalsePositiveRate = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SynthIdImageWatermarker<float>(new SynthIdImageOptions { ImageSize = Size, SecretBits = 16, SpecificityLossScale = -1 }));
    }
}
