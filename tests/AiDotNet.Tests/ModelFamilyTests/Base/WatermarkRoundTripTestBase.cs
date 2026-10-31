using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Safety.Watermarking;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// The shared contract of every keyed text and audio watermarker: what the key embeds, the same key finds; unmarked
/// content and other keys stay below the false-positive threshold; and the mark survives a mild perturbation.
/// </summary>
/// <remarks>
/// <para>
/// A watermarker's detection z-score is approximately standard normal on content without its key's mark, so the stated
/// false-positive rate of a single detection is P(Z &gt;= 3) = 0.135% (confidence 0 below z = 3). The tests check the
/// calibration directly: over <see cref="ForeignKeyCount"/> foreign keys the z-scores must look standard normal and at
/// most two may reach z = 3, which a calibrated detector exceeds with probability below 1e-4.
/// </para>
/// <para>
/// Every embed and detect call constructs a fresh watermarker from the key text, so a watermarker that draws a
/// random key per instance (marks nobody can find again) fails the round trip.
/// </para>
/// </remarks>
/// <typeparam name="TContent">The content type: token ids, text or audio samples.</typeparam>
public abstract class WatermarkRoundTripTestBase<TContent>
{
    /// <summary>The key the round-trip tests embed and detect with.</summary>
    protected const string TestKey = "round-trip-test-key-3c9f0a52";

    /// <summary>The number of foreign keys the calibration test detects with.</summary>
    protected virtual int ForeignKeyCount => 100;

    /// <summary>
    /// Whether the detector's z-score is standard normal on content without its mark. A detector that reports the best
    /// of several tests after a Bonferroni correction is calibrated only in its upper tail, so it returns <c>false</c>
    /// and the calibration test checks its false-positive count alone.
    /// </summary>
    protected virtual bool ReportsStandardNormalZ => true;

    /// <summary>The number of unmarked samples the false-positive test checks.</summary>
    protected virtual int UnmarkedSampleCount => 12;

    /// <summary>Unmarked content, deterministic in <paramref name="seed"/>.</summary>
    protected abstract TContent Sample(int seed);

    /// <summary>The marked version of <paramref name="content"/>, made by a fresh watermarker holding <paramref name="key"/>.</summary>
    protected abstract TContent Embed(string key, TContent content);

    /// <summary>The detection result of a fresh watermarker holding <paramref name="key"/>.</summary>
    protected abstract WatermarkResult Detect(string key, TContent content);

    /// <summary>A mild, realistic perturbation of <paramref name="content"/>, deterministic in <paramref name="seed"/>.</summary>
    protected abstract TContent Perturb(TContent content, int seed);

    /// <summary>Whether two contents are identical.</summary>
    protected abstract bool SameContent(TContent a, TContent b);

    [Fact(Timeout = 120000)]
    public async Task Embed_ThenDetect_WithTheSameKey_IsConfident()
    {
        await Task.Yield();
        for (int seed = 1; seed <= 3; seed++)
        {
            var result = Detect(TestKey, Embed(TestKey, Sample(seed)));
            Assert.True(result.ZScore >= 5.7 && result.Confidence >= 0.9,
                $"Sample {seed}: the mark scored z = {result.ZScore:F2} (confidence {result.Confidence:F3}) under its own key.");
            Assert.True(result.WatermarkDetected, $"Sample {seed}: a confident detection must be reported as detected.");
            Assert.True(result.PValue < 1e-8, $"Sample {seed}: p = {result.PValue:E2} for z = {result.ZScore:F2}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task UnmarkedContent_IsNotDetected()
    {
        await Task.Yield();
        for (int seed = 100; seed < 100 + UnmarkedSampleCount; seed++)
        {
            var result = Detect(TestKey, Sample(seed));
            Assert.True(result.ZScore < WatermarkStatistics.ZeroConfidenceZ,
                $"Unmarked sample {seed} scored z = {result.ZScore:F2}, above the false-positive threshold z = 3.");
            Assert.Equal(0.0, result.Confidence);
            Assert.False(result.WatermarkDetected);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task ForeignKeys_DoNotDetectTheMark_AtTheStatedFalsePositiveRate()
    {
        await Task.Yield();
        var marked = Embed(TestKey, Sample(7));
        var z = Enumerable.Range(0, ForeignKeyCount).Select(i => Detect($"foreign-key-{i}", marked).ZScore).ToArray();
        int falsePositives = z.Count(v => v >= WatermarkStatistics.ZeroConfidenceZ);
        double mean = z.Average();
        double sd = Math.Sqrt(z.Sum(v => (v - mean) * (v - mean)) / (z.Length - 1));

        Assert.True(falsePositives <= 2,
            $"{falsePositives} of {z.Length} foreign keys reached z = 3; a calibrated detector expects {0.00135 * z.Length:F2}.");
        if (ReportsStandardNormalZ)
        {
            Assert.True(Math.Abs(mean) < 0.4, $"Foreign-key z-scores average {mean:F3}; they should centre on 0.");
            Assert.True(sd > 0.6 && sd < 1.4, $"Foreign-key z-scores have standard deviation {sd:F3}; they should be near 1.");
        }
        Assert.True(Detect(TestKey, marked).ZScore >= 5.7, "The mark's own key must still detect it.");
    }

    [Fact(Timeout = 120000)]
    public async Task MildPerturbation_StillDetected()
    {
        await Task.Yield();
        for (int seed = 1; seed <= 2; seed++)
        {
            var perturbed = Perturb(Embed(TestKey, Sample(seed)), seed);
            var result = Detect(TestKey, perturbed);
            Assert.True(result.ZScore >= 4.5,
                $"Sample {seed}: after a mild perturbation the mark scored z = {result.ZScore:F2} (confidence {result.Confidence:F3}).");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Embedding_IsDeterministic_AndDependsOnTheKey()
    {
        await Task.Yield();
        var content = Sample(3);
        Assert.True(SameContent(Embed(TestKey, content), Embed(TestKey, content)),
            "Two watermarkers built from the same key must embed the same mark.");
        Assert.False(SameContent(Embed(TestKey, content), Embed("a-different-key", content)),
            "Different keys must embed different marks.");
    }
}
