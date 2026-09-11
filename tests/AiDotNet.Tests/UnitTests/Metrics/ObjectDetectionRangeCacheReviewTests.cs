using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.Metrics;

public sealed class ObjectDetectionRangeCacheReviewTests
{
    public enum ThresholdRange { Coco, TwoThresholds, OffGridEndpoint, SingleThreshold, IncludesZero, Batch31, Batch32, Batch33, Batch65 }
    public enum InvalidRange { ZeroStep, NegativeStep, NaNStep, PositiveInfiniteStep, NegativeInfiniteStep, NaNMinimum, InfiniteMinimum, NaNMaximum, InfiniteMaximum, NegativeMinimum, ExcessMaximum, Reversed, OverflowingCount, InfiniteCount }

    public ObjectDetectionRangeCacheReviewTests() => AiDotNet.Tests.TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(10)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(65)]
    public void Range_PreparesEachClassRankingOnceRegardlessOfThresholdCount(int thresholds)
    {
        var predictions = new CountingReadOnlyList<Detection<double>>(new[] { Det(0, 0, 10, 10, 0, 0.9) });
        var truth = new CountingReadOnlyList<Detection<double>>(new[] { Det(0, 0, 10, 10, 0, 1) });
        double result = new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(
            new[] { predictions }, new[] { truth }, 0.5, 0.5 + (thresholds - 1) / 128.0, 1.0 / 128);

        Assert.Equal(1.0, result);
        Assert.Equal(1, predictions.Enumerations);
        // One class-discovery pass and one ground-truth preparation pass, not two per threshold.
        Assert.Equal(2, truth.Enumerations);
    }

    [Fact]
    public void Range_RecomputesClaimsWhenTheEarlierPredictionOnlyClearsTheLowerThreshold()
    {
        var truth = OneImage(Det(0, 0, 10, 10, 0, 1));
        var predictions = OneImage(Det(0, 0, 10, 6, 0, 0.9), Det(0, 0, 10, 9, 0, 0.8));
        var metrics = new ObjectDetectionMetrics<double>();

        Assert.Equal(1.0, metrics.MeanAveragePrecision(predictions, truth, 0.5));
        Assert.Equal(0.5, metrics.MeanAveragePrecision(predictions, truth, 0.8));
        Assert.Equal(0.75, metrics.MeanAveragePrecisionRange(predictions, truth, 0.5, 0.8, 0.3));
    }

    [Fact]
    public void Range_EqualIoUTiesKeepTheFirstUnclaimedGroundTruthCandidate()
    {
        var truth = OneImage(Det(0, 0, 10, 10, 0, 1), Det(10, 0, 20, 10, 0, 1));
        var predictions = OneImage(Det(0, 0, 20, 10, 0, 0.9), Det(0, 0, 10, 10, 0, 0.8));
        var metrics = new ObjectDetectionMetrics<double>();

        Assert.Equal(51.0 / 101.0, metrics.MeanAveragePrecision(predictions, truth, 0.5), 12);
        Assert.Equal(0.5 * 51.0 / 101.0, metrics.MeanAveragePrecision(predictions, truth, 1), 12);
        Assert.Equal(0.75 * 51.0 / 101.0,
            metrics.MeanAveragePrecisionRange(predictions, truth, 0.5, 1, 0.5), 12);
    }

    [Theory]
    [InlineData(true, 1.0)]
    [InlineData(false, 0.5)]
    public void Range_PreservesStableConfidenceTies(bool correctPredictionFirst, double expected)
    {
        var correct = Det(0, 0, 10, 10, 0, 0.5);
        var wrong = Det(100, 100, 110, 110, 0, 0.5);
        var predictions = correctPredictionFirst ? OneImage(correct, wrong) : OneImage(wrong, correct);

        Assert.Equal(expected, new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(
            predictions, OneImage(Det(0, 0, 10, 10, 0, 1))));
    }

    [Theory]
    [InlineData(10)]
    [InlineData(33)]
    [InlineData(65)]
    public void Range_DoesNotReadAnUnusedMalformedBoxAfterEveryCandidateIsClaimed(int thresholds)
    {
        var unused = Det(0, 0, 10, 10, 0, 0.8);
        unused.Box.Format = BoundingBoxFormat.YOLO;
        Assert.Throws<InvalidOperationException>(() => unused.Box.ToXYXY());

        Assert.Equal(1.0, new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(
            OneImage(Det(0, 0, 10, 10, 0, 0.9), unused), OneImage(Det(0, 0, 10, 10, 0, 1)),
            0.5, 0.5 + (thresholds - 1) / 128.0, 1.0 / 128));
    }

    [Fact]
    public void Range_StillReadsAMalformedBoxWhenOnlyAHigherThresholdNeedsIt()
    {
        var needed = Det(0, 0, 10, 10, 0, 0.8);
        needed.Box.Format = BoundingBoxFormat.YOLO;
        Assert.Throws<InvalidOperationException>(() => new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(
            OneImage(Det(0, 0, 10, 6, 0, 0.9), needed), OneImage(Det(0, 0, 10, 10, 0, 1)), 0.5, 0.8, 0.3));
    }

    [Theory]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(65)]
    public void Range_BatchesPreserveUndefinedClassesAndZeroTruePositives(int thresholds)
    {
        var undefined = Det(0, 0, 10, 10, 1, 1);
        undefined.Box = (new BoundingBox<double>[1])[0];
        var truth = OneImage(Det(0, 0, 10, 10, 0, 1), undefined);
        var predictions = OneImage(Det(100, 100, 110, 110, 0, 0.9), Det(0, 0, 10, 10, 1, 0.8));
        var metrics = new ObjectDetectionMetrics<double>();
        double maximum = 0.5 + (thresholds - 1) / 128.0;

        Assert.True(double.IsNaN(metrics.AveragePrecision(predictions, truth, 1)));
        Assert.Equal(0.0, metrics.MeanAveragePrecisionRange(predictions, truth, 0.5, maximum, 1.0 / 128));
        Assert.Equal(0.0, metrics.MeanAveragePrecisionRange(predictions, OneImage(undefined), 0.5, maximum, 1.0 / 128));
        Assert.Equal(0.0, metrics.MeanAveragePrecisionRange(predictions, OneImage(), 0.5, maximum, 1.0 / 128));
        Assert.Equal(1.0, metrics.MeanAveragePrecisionRange(
            OneImage(Det(0, 0, 10, 10, 0, 0.9)), truth, 0.5, maximum, 1.0 / 128));
    }

    public static IEnumerable<object[]> InvalidCases()
    {
        foreach (InvalidRange range in Enum.GetValues(typeof(InvalidRange)))
            yield return new object[] { range };
    }

    [Theory]
    [MemberData(nameof(InvalidCases))]
    public void Range_RejectsNonFiniteOrUnrepresentableRangesBeforeReadingDetections(InvalidRange range)
    {
        var (minimum, maximum, step, parameter) = range switch
        {
            InvalidRange.ZeroStep => (0.5, 0.95, 0.0, "step"),
            InvalidRange.NegativeStep => (0.5, 0.95, -0.1, "step"),
            InvalidRange.NaNStep => (0.5, 0.95, double.NaN, "step"),
            InvalidRange.PositiveInfiniteStep => (0.5, 0.95, double.PositiveInfinity, "step"),
            InvalidRange.NegativeInfiniteStep => (0.5, 0.95, double.NegativeInfinity, "step"),
            InvalidRange.NaNMinimum => (double.NaN, 0.95, 0.05, "minIoU"),
            InvalidRange.InfiniteMinimum => (double.NegativeInfinity, 0.95, 0.05, "minIoU"),
            InvalidRange.NaNMaximum => (0.5, double.NaN, 0.05, "minIoU"),
            InvalidRange.InfiniteMaximum => (0.5, double.PositiveInfinity, 0.05, "minIoU"),
            InvalidRange.NegativeMinimum => (-0.1, 0.95, 0.05, "minIoU"),
            InvalidRange.ExcessMaximum => (0.5, 1.1, 0.05, "minIoU"),
            InvalidRange.Reversed => (0.95, 0.5, 0.05, "minIoU"),
            InvalidRange.OverflowingCount => (0.0, 1.0, 1.0 / int.MaxValue, "step"),
            InvalidRange.InfiniteCount => (0.0, 1.0, double.Epsilon, "step"),
            _ => throw new ArgumentOutOfRangeException(nameof(range))
        };
        var predictions = new CountingReadOnlyList<Detection<double>>(Array.Empty<Detection<double>>());
        var truth = new CountingReadOnlyList<Detection<double>>(Array.Empty<Detection<double>>());
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(
            new[] { predictions }, new[] { truth }, minimum, maximum, step));

        Assert.Equal(parameter, error.ParamName);
        Assert.Equal(0, predictions.Enumerations);
        Assert.Equal(0, truth.Enumerations);
    }

    [Fact]
    public void Range_AcceptsSingleThresholdWithTheSmallestPositiveStep()
        => Assert.Equal(1.0, new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(
            OneImage(Det(0, 0, 10, 10, 0, 0.9)), OneImage(Det(0, 0, 10, 10, 0, 1)), 0.5, 0.5, double.Epsilon));

    [Fact]
    public void Range_ZeroOverlapIsNotAMatchEvenAtZeroThreshold()
    {
        Assert.Equal(0.0, new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(
            OneImage(Det(100, 100, 110, 110, 0, 0.9)), OneImage(Det(0, 0, 10, 10, 0, 1)), 0, 1, 0.1));
    }

    [Fact]
    public void Range_PreservesNullImageAndNullDetectionFiltering()
    {
        // Array initialization models null values received from external callers without suppressing analysis.
        var predictions = new IReadOnlyList<Detection<double>>[3];
        var truth = new IReadOnlyList<Detection<double>>[3];
        predictions[1] = new Detection<double>[1];
        truth[1] = new Detection<double>[1];
        predictions[2] = new[] { Det(0, 0, 10, 10, 0, 0.9) };
        truth[2] = new[] { Det(0, 0, 10, 10, 0, 1) };

        Assert.Equal(1.0, new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(predictions, truth));
    }

    public static IEnumerable<object[]> DeterministicCases()
    {
        foreach (int seed in new[] { 1, 1337, 8291 })
            foreach (ThresholdRange range in Enum.GetValues(typeof(ThresholdRange)))
                yield return new object[] { seed, range };
    }

    [Theory]
    [MemberData(nameof(DeterministicCases))]
    public void Range_IsExactlyTheOrderedMeanOfIndependentPerThresholdScores(int seed, ThresholdRange range)
    {
        var random = new Random(seed);
        var predictions = new List<IReadOnlyList<Detection<double>>>();
        var truth = new List<IReadOnlyList<Detection<double>>>();
        for (int image = 0; image < 4; image++)
        {
            var actual = new List<Detection<double>>();
            var predicted = new List<Detection<double>>();
            for (int index = 0; index < 12; index++)
            {
                double x = index % 4 * 8;
                double y = index / 4 * 8;
                actual.Add(Det(x, y, x + 10, y + 10, index % 3, 1));
                predicted.Add(Det(x, y, x + 10, y + 5 + random.NextDouble() * 5,
                    index % 3, random.Next(4) / 4.0));
                predicted.Add(Det(x + 2, y + 1, x + 12, y + 11,
                    random.Next(4), random.Next(4) / 4.0));
            }
            truth.Add(actual);
            predictions.Add(predicted);
        }

        var (minimum, maximum, step) = range switch
        {
            ThresholdRange.Coco => (0.5, 0.95, 0.05),
            ThresholdRange.TwoThresholds => (0.5, 0.8, 0.3),
            ThresholdRange.OffGridEndpoint => (0.4, 0.95, 0.2),
            ThresholdRange.SingleThreshold => (0.8, 0.8, 0.05),
            ThresholdRange.IncludesZero => (0.0, 1.0, 0.1),
            ThresholdRange.Batch31 => (0.5, 0.5 + 30.0 / 128, 1.0 / 128),
            ThresholdRange.Batch32 => (0.5, 0.5 + 31.0 / 128, 1.0 / 128),
            ThresholdRange.Batch33 => (0.5, 0.5 + 32.0 / 128, 1.0 / 128),
            ThresholdRange.Batch65 => (0.5, 1.0, 1.0 / 128),
            _ => throw new ArgumentOutOfRangeException(nameof(range))
        };
        var metrics = new ObjectDetectionMetrics<double>();
        int count = (int)Math.Floor((maximum - minimum) / step + 1e-9) + 1;
        double expected = 0;
        for (int index = 0; index < count; index++)
            expected += metrics.MeanAveragePrecision(predictions, truth, minimum + index * step);

        // Exact equality also guards the order of summing per-class and per-threshold scores.
        Assert.Equal(expected / count, metrics.MeanAveragePrecisionRange(predictions, truth, minimum, maximum, step));
    }

    private static Detection<double> Det(double x1, double y1, double x2, double y2, int classId, double confidence)
        => new(new BoundingBox<double>(x1, y1, x2, y2), classId, confidence);

    private static IReadOnlyList<IReadOnlyList<Detection<double>>> OneImage(params Detection<double>[] detections)
        => new[] { detections };

    private sealed class CountingReadOnlyList<TItem> : IReadOnlyList<TItem>
    {
        private readonly IReadOnlyList<TItem> _items;
        public CountingReadOnlyList(IReadOnlyList<TItem> items) => _items = items;
        public int Count => _items.Count;
        public TItem this[int index] => _items[index];
        public int Enumerations { get; private set; }
        public IEnumerator<TItem> GetEnumerator()
        {
            Enumerations++;
            return _items.GetEnumerator();
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
