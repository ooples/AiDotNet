using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ComputerVision.Segmentation.Video;
using AiDotNet.Interfaces;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ComputerVision;

public sealed class DevaVideoOptionFixtureTests
{
    private static double Aggregate(double probability) => probability * probability /
        (probability * probability + (1 - probability) * (1 - probability));

    private static double[] Values(IReadOnlyList<VideoSegmentationResult<double>> results) =>
        results.SelectMany(result => result.ObjectIds.Select(id => (double)id).Prepend(result.ObjectIds.Length)
            .Concat(result.Masks.ToArray()).Concat(result.Confidences)).ToArray();

    [Theory]
    [InlineData(15)]
    [InlineData(30)]
    public void TopK_public_video_read_matches_scalar_negative_L2_softmax(int topK)
    {
        var result = Assert.Single(DevaOptionFixture.Run<double>(new DEVAOptions { MemoryTopK = topK }, "MemoryTopK"));
        var selected = Enumerable.Range(0, 64).Select(i => new
        {
            Index = i, Affinity = (.4 * (i * .03) - Math.Pow(i * .03, 2)) / Math.Sqrt(2),
        }).OrderByDescending(item => item.Affinity).ThenBy(item => item.Index).Take(topK).ToArray();
        double maximum = selected.Max(item => item.Affinity);
        double total = selected.Sum(item => Math.Exp(item.Affinity - maximum));
        double read = selected.Sum(item => Math.Exp(item.Affinity - maximum) * (.1 + .8 * item.Index / 63)) / total;
        Assert.Equal(new[] { 7 }, result.ObjectIds);
        Assert.All(result.Masks.ToArray(), value => Assert.Equal(Aggregate(read), value, 10));
    }

    [Theory]
    [InlineData(9)]
    [InlineData(10)]
    public void Memory_eviction_keeps_annotation_and_newest_frames(int capacity)
    {
        var results = DevaOptionFixture.Run<double>(new DEVAOptions { MaxMemoryFrames = capacity }, "MaxMemoryFrames");
        var retained = new List<double> { .1 };
        for (int step = 1; step <= 12; step++)
        {
            double correction = .1 + .05 * step;
            retained.Add(correction);
            if (retained.Count > capacity) retained.RemoveAt(1);
            Assert.All(results[step - 1].Masks.ToArray(), value => Assert.Equal(correction, value, 12));
        }
        Assert.All(results[12].Masks.ToArray(), value => Assert.Equal(Aggregate(retained.Average()), value, 10));
    }

    [Theory]
    [InlineData("MergeEvery", 4)]
    [InlineData("MergeEvery", 5)]
    [InlineData("MaxMissedMerges", 4)]
    [InlineData("MaxMissedMerges", 5)]
    public void Background_video_deletes_only_at_configured_unmatched_integration(string option, int boundary)
    {
        var options = new DEVAOptions { MergeEvery = boundary, MaxMissedMerges = boundary };
        var results = DevaOptionFixture.Run<double>(options, option);
        for (int index = 0; index < results.Count; index++)
        {
            if (index + 1 < boundary) Assert.Equal(new[] { 7 }, results[index].ObjectIds);
            else Assert.Empty(results[index].ObjectIds);
        }
    }

    [Theory]
    [InlineData(.49, 1)]
    [InlineData(.5, 2)]
    public void Association_uses_strict_IoU_of_half_image_against_full_annotation(double threshold, int objects)
    {
        var result = Assert.Single(DevaOptionFixture.Run<double>(
            new DEVAOptions { AssociationIoUThreshold = threshold }, "AssociationIoUThreshold"));
        Assert.Equal(objects, result.ObjectIds.Length);
        Assert.Equal(7, result.ObjectIds[0]);
        if (objects == 1) Assert.All(result.Masks.ToArray(), value => Assert.Equal(1, value));
        else Assert.Equal(8, result.ObjectIds[1]);
    }

    [Theory]
    [InlineData("MemoryTopK")]
    [InlineData("MergeEvery")]
    [InlineData("MaxMissedMerges")]
    [InlineData("AssociationIoUThreshold")]
    [InlineData("MaxMemoryFrames")]
    public void Each_temporal_option_changes_actual_video_results_and_repeated_control_is_deterministic(string option)
    {
        var control = new DEVAOptions();
        var candidate = new DEVAOptions(control);
        switch (option)
        {
            case "MemoryTopK": candidate.MemoryTopK = 15; break;
            case "MergeEvery": candidate.MergeEvery = 4; break;
            case "MaxMissedMerges": candidate.MaxMissedMerges = 4; break;
            case "AssociationIoUThreshold": candidate.AssociationIoUThreshold = .25; break;
            case "MaxMemoryFrames": candidate.MaxMemoryFrames = 9; break;
        }
        var before = Values(DevaOptionFixture.Run<double>(control, option));
        var repeat = Values(DevaOptionFixture.Run<double>(control, option));
        var changed = Values(DevaOptionFixture.Run<double>(candidate, option));
        Assert.Equal(before, repeat);
        Assert.False(before.SequenceEqual(changed));
        // The fixture copies configurations: observing a task must not rewrite the original model's options.
        Assert.Equal(30, control.MemoryTopK);
        Assert.Equal(5, control.MergeEvery);
        Assert.Equal(5, control.MaxMissedMerges);
        Assert.Equal(.5, control.AssociationIoUThreshold);
        Assert.Equal(10, control.MaxMemoryFrames);
    }
}
