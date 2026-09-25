using System.Collections;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.Metrics;

public sealed class ObjectDetectionThresholdBoundaryReviewTests
{
    public enum MetricEntry { AveragePrecision, MeanAveragePrecision, PrecisionRecallCurve }
    public enum InvalidThreshold { NaN, NegativeInfinity, PositiveInfinity, BelowZero, AboveOne }
    public enum DataShape { Populated, EmptyImages, EmptyGroundTruth }
    public enum GridBoundary { BelowUnitEndpoint, BelowInteriorEndpoint, ExactUnitEndpoint, DecimalEndpoint, Coco, OffGridEndpoint, SingleThreshold }

    public ObjectDetectionThresholdBoundaryReviewTests() => AiDotNet.Tests.TestModuleInitializer.EnsureInitialized();

    public static IEnumerable<object[]> InvalidCases()
    {
        foreach (MetricEntry entry in Enum.GetValues(typeof(MetricEntry)))
            foreach (InvalidThreshold invalid in Enum.GetValues(typeof(InvalidThreshold)))
                foreach (DataShape shape in Enum.GetValues(typeof(DataShape)))
                    yield return new object[] { entry, invalid, shape };
    }

    [Theory]
    [MemberData(nameof(InvalidCases))]
    public void SingleThreshold_RejectsNonFiniteOrOutOfRangeValuesBeforeEnumeratingData(
        MetricEntry entry, InvalidThreshold invalid, DataShape shape)
    {
        double threshold = InvalidValue(invalid);
        var predictions = new CountingList(shape == DataShape.EmptyImages ? Array.Empty<Detection<double>>() : new[] { Box(1) });
        var truth = new CountingList(shape == DataShape.Populated ? new[] { Box(1) } : Array.Empty<Detection<double>>());
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => Evaluate(entry, new[] { predictions }, new[] { truth }, threshold));
        Assert.Equal("iouThreshold", error.ParamName);
        Assert.Equal(threshold, Assert.IsType<double>(error.ActualValue));
        Assert.Equal(0, predictions.Enumerations);
        Assert.Equal(0, truth.Enumerations);
    }

    [Theory]
    [InlineData(MetricEntry.AveragePrecision, 0.0)]
    [InlineData(MetricEntry.AveragePrecision, 1.0)]
    [InlineData(MetricEntry.MeanAveragePrecision, 0.0)]
    [InlineData(MetricEntry.MeanAveragePrecision, 1.0)]
    [InlineData(MetricEntry.PrecisionRecallCurve, 0.0)]
    [InlineData(MetricEntry.PrecisionRecallCurve, 1.0)]
    public void SingleThreshold_AcceptsInclusiveFiniteEndpoints(MetricEntry entry, double threshold)
        => Assert.Equal(1.0, Evaluate(entry, OneImage(Box(1)), OneImage(Box(1)), threshold));

    [Theory]
    [InlineData(MetricEntry.AveragePrecision)]
    [InlineData(MetricEntry.MeanAveragePrecision)]
    [InlineData(MetricEntry.PrecisionRecallCurve)]
    public void SingleThreshold_PreservesNullAndAlignmentErrorPrecedence(MetricEntry entry)
    {
        var missing = new IReadOnlyList<IReadOnlyList<Detection<double>>>[1];
        var nullError = Assert.Throws<ArgumentNullException>(() => Evaluate(entry, missing[0], OneImage(), double.NaN));
        Assert.Equal("predictions", nullError.ParamName);
        Assert.Throws<ArgumentException>(() => Evaluate(entry, Array.Empty<IReadOnlyList<Detection<double>>>(), OneImage(), double.NaN));
    }

    [Theory]
    [InlineData(GridBoundary.BelowUnitEndpoint)]
    [InlineData(GridBoundary.BelowInteriorEndpoint)]
    [InlineData(GridBoundary.ExactUnitEndpoint)]
    [InlineData(GridBoundary.DecimalEndpoint)]
    [InlineData(GridBoundary.Coco)]
    [InlineData(GridBoundary.OffGridEndpoint)]
    [InlineData(GridBoundary.SingleThreshold)]
    public void Range_UsesOnlyItsGridAndNeverEvaluatesAboveTheMaximum(GridBoundary boundary)
    {
        var (minimum, maximum, step, overlap, expected) = boundary switch
        {
            GridBoundary.BelowUnitEndpoint => (0.0, 0.9999999995, 1.0, 0.5, 1.0),
            GridBoundary.BelowInteriorEndpoint => (0.125, 0.8749999995, 0.75, 0.5, 1.0),
            GridBoundary.ExactUnitEndpoint => (0.0, 1.0, 1.0, 0.5, 0.5),
            // Binary .1 + 2*.1 exceeds the supplied .3 by one ULP. An inclusive on-grid
            // endpoint must be scored at max itself, never at a larger reconstructed value.
            GridBoundary.DecimalEndpoint => (0.1, 0.3, 0.1, 0.3, 1.0),
            GridBoundary.Coco => (0.5, 0.95, 0.05, 0.925, 0.9),
            GridBoundary.OffGridEndpoint => (0.4, 0.95, 0.2, 0.9, 1.0),
            GridBoundary.SingleThreshold => (0.3, 0.3, double.Epsilon, 0.3, 1.0),
            _ => throw new ArgumentOutOfRangeException(nameof(boundary))
        };
        var metrics = new ObjectDetectionMetrics<double>();
        var predictions = OneImage(Box(overlap));
        var truth = OneImage(Box(1));
        Assert.Equal(overlap, predictions[0][0].Box.IoU(truth[0][0].Box), 12);
        Assert.Equal(expected, metrics.MeanAveragePrecisionRange(predictions, truth, minimum, maximum, step), 12);
    }

    [Fact]
    public void Range_DoesNotImposeAnArbitraryThresholdCountCapOnEmptyData()
    {
        // The count is Int32.MaxValue, but no matching state is allocated for no classes.
        Assert.Equal(0.0, new ObjectDetectionMetrics<double>().MeanAveragePrecisionRange(
            OneImage(), OneImage(), 0, 1, 1.0 / (int.MaxValue - 1)));
    }

    private static double Evaluate(MetricEntry entry, IReadOnlyList<IReadOnlyList<Detection<double>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<double>>> truth, double threshold)
    {
        var metrics = new ObjectDetectionMetrics<double>();
        return entry switch
        {
            MetricEntry.AveragePrecision => metrics.AveragePrecision(predictions, truth, 0, threshold),
            MetricEntry.MeanAveragePrecision => metrics.MeanAveragePrecision(predictions, truth, threshold),
            MetricEntry.PrecisionRecallCurve => Assert.Single(metrics.PrecisionRecallCurve(predictions, truth, 0, threshold).Precision),
            _ => throw new ArgumentOutOfRangeException(nameof(entry))
        };
    }

    private static double InvalidValue(InvalidThreshold invalid) => invalid switch
    {
        InvalidThreshold.NaN => double.NaN, InvalidThreshold.NegativeInfinity => double.NegativeInfinity,
        InvalidThreshold.PositiveInfinity => double.PositiveInfinity, InvalidThreshold.BelowZero => -double.Epsilon,
        InvalidThreshold.AboveOne => 1.000000000000001,
        _ => throw new ArgumentOutOfRangeException(nameof(invalid))
    };

    private static Detection<double> Box(double width) => new(new BoundingBox<double>(0, 0, width, 1), 0, 0.9);
    private static IReadOnlyList<IReadOnlyList<Detection<double>>> OneImage(params Detection<double>[] detections) => new[] { detections };

    private sealed class CountingList : IReadOnlyList<Detection<double>>
    {
        private readonly IReadOnlyList<Detection<double>> _items;
        public CountingList(IReadOnlyList<Detection<double>> items) => _items = items;
        public int Count => _items.Count;
        public Detection<double> this[int index] => _items[index];
        public int Enumerations { get; private set; }
        public IEnumerator<Detection<double>> GetEnumerator() { Enumerations++; return _items.GetEnumerator(); }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
