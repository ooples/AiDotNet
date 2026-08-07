using System;
using System.Collections.Generic;
using System.Linq;
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
        var frames = Frames(120);
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
        var frames = Frames(60);
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
        }
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
