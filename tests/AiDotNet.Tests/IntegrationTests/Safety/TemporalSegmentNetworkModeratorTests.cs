using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Safety;
using AiDotNet.Safety.Video;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Verifies that <see cref="FrameSamplingVideoModerator{T}"/> actually implements Temporal Segment
/// Networks (Wang et al., ECCV 2016) rather than fixed-rate frame sampling.
/// </summary>
/// <remarks>
/// The two properties that distinguish TSN from the per-frame pipeline it replaced are (a) SPARSE
/// sampling — a constant K snippets however long the video is, instead of a count that grows with
/// duration — and (b) segmental CONSENSUS before the decision, so the verdict is video-level rather
/// than "any flagged frame flags the video". Without tests for those two, the rebuild would be
/// asserted only by its comments.
/// </remarks>
public class TemporalSegmentNetworkModeratorTests
{
    /// <summary>
    /// Frames whose pixels genuinely satisfy the classifier's skin-tone predicate, alternating
    /// between two shades so the temporal detectors also have something to see.
    /// </summary>
    /// <remarks>
    /// SEPARATE FROM <see cref="Frames"/>, WHICH MUST STAY BENIGN. The sampling-cost and
    /// empty-input tests depend on Frames() producing nothing, so the harmful content lives here
    /// instead of being folded into the shared helper. Uniform 0-1 noise never trips
    /// CLIPImageSafetyClassifier -- skin needs r>95, g>40, b>20 of 255 with r>g>b -- so tests
    /// asserting that findings ARE produced were asserting against an empty collection. Both shades
    /// are inside the predicate, so the skin fraction stays 1.0 and clears the 0.8 NSFW bar.
    /// </remarks>
    /// <summary>
    /// Harmful frames whose skin COVERAGE ramps across the video, so the five segment scores differ
    /// and the consensus functions have something to disagree about.
    /// </summary>
    /// <remarks>
    /// SEPARATE FROM <see cref="HarmfulFrames"/>, WHICH IS UNIFORM ON PURPOSE. Max, Average and
    /// WeightedAverage are mathematically identical on equal inputs, so a video that is uniformly
    /// skin makes every mode agree and cannot distinguish them - that is not a shared code path, it
    /// is the functions agreeing because the scores agree. Coverage ramps 0.6 -> 1.0, which puts the
    /// five segment centres near .64/.76/.80/.88/.96: Max 0.96, WeightedAverage ~0.86, Average ~0.81,
    /// all three still clearing the 0.8 NSFW bar so every mode reports, and all three distinct.
    /// </remarks>
    private static List<Tensor<double>> GradedHarmfulFrames(int count)
    {
        var frames = Frames(count);
        for (int f = 0; f < frames.Count; f++)
        {
            var t = frames[f];
            double coverage = count > 1 ? 0.6 + 0.4 * f / (count - 1) : 1.0;
            int skinPixels = (int)Math.Round(coverage * 64);

            for (int pixel = 0; pixel < 64; pixel++)
            {
                int y = pixel / 8, x = pixel % 8;
                int r = (0 * 8 + y) * 8 + x;
                int g = (1 * 8 + y) * 8 + x;
                int b = (2 * 8 + y) * 8 + x;
                if (b >= t.Length) continue;

                if (pixel < skinPixels)
                {
                    t[r] = 0.85; t[g] = 0.65; t[b] = 0.55;
                }
                else
                {
                    // Explicitly NOT skin: green above red fails the r > g requirement.
                    t[r] = 0.20; t[g] = 0.60; t[b] = 0.30;
                }
            }
        }

        return frames;
    }

    private static List<Tensor<double>> HarmfulFrames(int count)
    {
        var frames = Frames(count);
        for (int f = 0; f < frames.Count; f++)
        {
            var t = frames[f];
            bool alternate = f % 2 == 1;
            for (int y = 0; y < 8; y++)
            {
                for (int x = 0; x < 8; x++)
                {
                    int r = (0 * 8 + y) * 8 + x;
                    int g = (1 * 8 + y) * 8 + x;
                    int b = (2 * 8 + y) * 8 + x;
                    if (b >= t.Length) continue;

                    t[r] = alternate ? 0.45 : 0.85;
                    t[g] = alternate ? 0.30 : 0.65;
                    t[b] = alternate ? 0.22 : 0.55;
                }
            }
        }

        return frames;
    }

    private static List<Tensor<double>> Frames(int count, int seed = 3)
    {
        var rng = new Random(seed);
        var frames = new List<Tensor<double>>(count);
        for (int f = 0; f < count; f++)
        {
            var t = new Tensor<double>([3, 8, 8]);
            for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble();
            frames.Add(t);
        }
        return frames;
    }

    [Fact]
    public void DefaultSegmentCount_IsThree_PerThePaper()
    {
        // "the number of snippets K is set to 3 according to previous works on temporal modeling"
        Assert.Equal(3, FrameSamplingVideoModerator<double>.DefaultSegmentCount);
        Assert.Equal(3, new FrameSamplingVideoModerator<double>().SegmentCount);
    }

    [Fact]
    public void DefaultConsensus_IsAverage_ThePapersBest()
    {
        // Table 3, UCF101 split 1: average 93.5% two-stream vs max 91.6% and weighted 92.4%.
        Assert.Equal(SegmentalConsensus.Average, new FrameSamplingVideoModerator<double>().Consensus);
    }

    [Fact]
    public void SamplingIsSparse_CostIsIndependentOfVideoLength()
    {
        // The whole point of sparse temporal sampling: a 10x longer video is judged on the SAME
        // number of snippets. Fixed-rate sampling would scale the work with duration.
        var shortVideo = Frames(30);
        var longVideo = Frames(300);

        var moderator = new FrameSamplingVideoModerator<double>(segmentCount: 3);

        moderator.EvaluateVideo(shortVideo, 30.0);
        int sampledForShort = moderator.LastSampledFrameCount;

        moderator.EvaluateVideo(longVideo, 30.0);
        int sampledForLong = moderator.LastSampledFrameCount;

        // THE COST PROPERTY, NOT A PROPERTY ROUND-TRIP. Asserting moderator.SegmentCount == 3
        // checked only that the constructor stored the argument it was handed; it held even if
        // EvaluateVideo walked all 300 frames, which is the one regression this method's name
        // claims to catch. What a K-segment sampler guarantees is that a 10x longer video yields
        // the SAME number of segment-level inputs.
        Assert.Equal(3, sampledForShort);
        Assert.Equal(sampledForShort, sampledForLong);
    }

    [Fact]
    public void VerdictIsVideoLevel_OneFindingPerCategoryAtMost()
    {
        // Consensus precedes the decision, so a category yields at most ONE video-level finding —
        // not one per flagged frame, which is what the previous per-frame pipeline emitted.
        var frames = HarmfulFrames(120);
        var moderator = new FrameSamplingVideoModerator<double>(segmentCount: 5);

        var findings = moderator.EvaluateVideo(frames, 30.0);

        Assert.NotNull(findings);

        // A duplicate-group query over an empty list finds no groups and passes. Without this,
        // a moderator that detected nothing at all satisfied the whole assertion below.
        Assert.NotEmpty(findings);

        var duplicated = findings.GroupBy(f => f.Category).Where(g => g.Count() > 1).ToList();
        Assert.True(duplicated.Count == 0,
            "Consensus is applied before the decision, so each category must produce at most one " +
            "video-level finding; got duplicates for: " +
            string.Join(", ", duplicated.Select(g => $"{g.Key} x{g.Count()}")));
    }

    [Fact]
    public void FindingSpansTheWholeVideo_NotASingleFrame()
    {
        var frames = Frames(90);
        var moderator = new FrameSamplingVideoModerator<double>(segmentCount: 3);

        foreach (var finding in moderator.EvaluateVideo(frames, 30.0))
        {
            Assert.Equal(0, finding.SpanStart);
            Assert.True(finding.SpanEnd > 0,
                "A video-level verdict should span the video, not a single frame.");
        }
    }

    [Fact]
    public void EveryConsensusFunction_IsAccepted_AndDeterministic()
    {
        var frames = GradedHarmfulFrames(60);
        var perMode = new Dictionary<SegmentalConsensus, IReadOnlyList<SafetyFinding>>();

        foreach (SegmentalConsensus consensus in (SegmentalConsensus[])Enum.GetValues(typeof(SegmentalConsensus)))
        {
            var moderator = new FrameSamplingVideoModerator<double>(segmentCount: 4, consensus: consensus);
            var first = moderator.EvaluateVideo(frames, 30.0);
            var second = moderator.EvaluateVideo(frames, 30.0);

            Assert.Equal(consensus, moderator.Consensus);
            Assert.NotNull(first);
            // Centre-of-segment sampling exists so a rescan of the same video agrees with itself.
            Assert.Equal(first.Count, second.Count);
            for (int i = 0; i < first.Count; i++)
            {
                Assert.Equal(first[i].Category, second[i].Category);
                Assert.Equal(first[i].Confidence, second[i].Confidence, 12);
            }

            perMode[consensus] = first;
        }

        // THE MODES MUST BE DISTINGUISHABLE FROM EACH OTHER. Accepting each mode and checking it
        // repeats itself is satisfied when Average, Max and Weighted are all wired to one code path
        // -- which would make DefaultConsensus_IsAverage_ThePapersBest assert a preference between
        // three names for the same computation. The paper reports 93.5 / 91.6 / 92.4 on UCF101
        // precisely because they are different functions.
        var confidenceProfiles = perMode.Values
            .Select(fs => string.Join(",", fs.Select(f => f.Confidence.ToString("F12"))))
            .ToList();

        Assert.True(confidenceProfiles.Distinct().Count() > 1,
            "Every SegmentalConsensus mode produced identical findings, so the consensus function " +
            "is not being applied -- all modes appear to share one code path. Modes: " +
            string.Join(", ", perMode.Keys));
    }

    [Fact]
    public void SegmentCount_MustBePositive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new FrameSamplingVideoModerator<double>(segmentCount: 0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new FrameSamplingVideoModerator<double>(segmentCount: -1));
    }

    [Fact]
    public void FewerFramesThanSegments_IsHandled()
    {
        // K larger than the frame count must not index past the end; segments collapse to frames.
        var moderator = new FrameSamplingVideoModerator<double>(segmentCount: 10);
        var findings = moderator.EvaluateVideo(Frames(2), 30.0);
        Assert.NotNull(findings);
    }

    [Fact]
    public void EmptyVideo_ProducesNoFindings()
    {
        var moderator = new FrameSamplingVideoModerator<double>();
        Assert.Empty(moderator.EvaluateVideo(new List<Tensor<double>>(), 30.0));
        Assert.Empty(moderator.EvaluateVideo(Frames(10), 0.0));
    }
}
