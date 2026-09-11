using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Metrics;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for text detectors (CRAFT, DBNet, EAST).
/// </summary>
/// <remarks>
/// <para>
/// Text detectors localise words and lines as polygons rather than axis-aligned boxes, because
/// scene and document text is routinely rotated or curved. The ICDAR evaluation protocol scores
/// polygon overlap, so these invariants pin what a scorable polygon looks like: enough vertices
/// to enclose area, a positive area once enclosed, finite coordinates, and a bounding box that
/// actually bounds it. A region failing any of these is silently unscoreable.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type the detector is expressed in.</typeparam>
public abstract class TextDetectionTestBase<T> : DetectionModelTestBase<T>
    where T : struct
{
    /// <summary>
    /// The model under test as a text detector. Family resolution guarantees the cast.
    /// </summary>
    protected TextDetectorBase<T> CreateTextDetector() => (TextDetectorBase<T>)CreateModel();

    /// <summary>Confidence threshold used when the test does not vary it.</summary>
    protected virtual double DetectConfidenceThreshold => 0.05;

    private List<(double X, double Y)> PolygonOf(TextRegion<T> region)
    {
        var polygon = new List<(double X, double Y)>();
        if (region.Polygon is not null)
        {
            foreach (var vertex in region.Polygon)
            {
                polygon.Add((ToD(vertex.X), ToD(vertex.Y)));
            }
        }

        return polygon;
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_PolygonsShouldEncloseArea()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();

        var result = detector.Detect(CreateRandomImage(rng), DetectConfidenceThreshold);

        Assert.NotNull(result);
        Assert.NotNull(result.TextRegions);
        foreach (var region in result.TextRegions)
        {
            var polygon = PolygonOf(region);
            if (polygon.Count == 0)
            {
                continue; // Box-only region; covered by the box invariant below.
            }

            Assert.True(
                polygon.Count >= 4,
                $"Text polygon has only {polygon.Count} vertices; a quadrilateral is the minimum "
                + "the ICDAR protocol accepts.");

            foreach (var (x, y) in polygon)
            {
                Assert.False(double.IsNaN(x) || double.IsNaN(y), "Text polygon has a NaN vertex.");
                Assert.False(double.IsInfinity(x) || double.IsInfinity(y),
                    "Text polygon has an infinite vertex.");
            }

            Assert.True(
                TextDetectionMetrics<T>.PolygonArea(polygon) > 0.0,
                "Text polygon encloses zero area, so every IoU against it is zero and the region "
                + "can never be scored as a match.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_BoxesShouldBeGeometricallyValid()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();

        var result = detector.Detect(CreateRandomImage(rng), DetectConfidenceThreshold);

        foreach (var region in result.TextRegions)
        {
            Assert.NotNull(region.Box);
            var (xMin, yMin, xMax, yMax) = region.Box.ToXYXY();

            Assert.False(double.IsNaN(xMin) || double.IsNaN(yMin) || double.IsNaN(xMax) || double.IsNaN(yMax),
                "Text region box has a NaN coordinate.");
            Assert.True(xMax > xMin, $"Text region box has inverted or zero width: {xMin} to {xMax}.");
            Assert.True(yMax > yMin, $"Text region box has inverted or zero height: {yMin} to {yMax}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_BoxShouldBoundItsPolygon()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();

        var result = detector.Detect(CreateRandomImage(rng), DetectConfidenceThreshold);

        foreach (var region in result.TextRegions)
        {
            var polygon = PolygonOf(region);
            if (polygon.Count == 0)
            {
                continue;
            }

            var (xMin, yMin, xMax, yMax) = region.Box.ToXYXY();

            // Consumers that cannot handle polygons fall back to the box. If the box does not
            // contain the polygon, that fallback silently crops the detected word.
            foreach (var (x, y) in polygon)
            {
                Assert.InRange(x, xMin - 1e-6, xMax + 1e-6);
                Assert.InRange(y, yMin - 1e-6, yMax + 1e-6);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ConfidencesShouldBeInUnitRange()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();

        var result = detector.Detect(CreateRandomImage(rng), DetectConfidenceThreshold);

        foreach (var region in result.TextRegions)
        {
            Assert.InRange(ToD(region.Confidence), 0.0, 1.0);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ShouldRespectMaxDetections()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();

        var result = detector.Detect(CreateRandomImage(rng), 0.0);

        Assert.True(
            result.TextRegions.Count <= detector.MaxDetections,
            $"Detector returned {result.TextRegions.Count} regions, above its MaxDetections "
            + $"of {detector.MaxDetections}.");
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_RaisingTheConfidenceThresholdCannotAddRegions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();
        var image = CreateRandomImage(rng);

        int lenient = detector.Detect(image, 0.05).TextRegions.Count;
        int strict = detector.Detect(image, 0.9).TextRegions.Count;

        Assert.True(
            strict <= lenient,
            $"Raising the confidence threshold increased the region count: {lenient} at 0.05, "
            + $"{strict} at 0.9.");
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ShouldReportTheSourceImageDimensions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();

        var image = CreateRandomImage(rng);
        var result = detector.Detect(image, DetectConfidenceThreshold);

        Assert.Equal(image.Shape[3], result.ImageWidth);
        Assert.Equal(image.Shape[2], result.ImageHeight);
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();
        var image = CreateRandomImage(rng);

        var first = detector.Detect(image, DetectConfidenceThreshold).TextRegions;
        var second = detector.Detect(image, DetectConfidenceThreshold).TextRegions;

        Assert.Equal(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
        {
            Assert.Equal(ToD(first[i].Confidence), ToD(second[i].Confidence), 10);

            var (x1, y1, x2, y2) = first[i].Box.ToXYXY();
            var (u1, v1, u2, v2) = second[i].Box.ToXYXY();
            Assert.Equal(x1, u1, 8);
            Assert.Equal(y1, v1, 8);
            Assert.Equal(x2, u2, 8);
            Assert.Equal(y2, v2, 8);
        }
    }
}

/// <summary>Default-precision alias used by the generated fixtures.</summary>
public abstract class TextDetectionTestBase : TextDetectionTestBase<double> { }
