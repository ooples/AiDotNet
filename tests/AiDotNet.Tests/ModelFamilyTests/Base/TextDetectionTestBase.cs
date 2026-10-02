using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Metrics;
using AiDotNet.Models.Options;
using AiDotNet.Models.Parameters;
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

    /// <summary>Generated factory for a bounded, controlled positive fixture; normal defaults stay unchanged.</summary>
    protected abstract TextDetectorBase<T> CreatePositiveTextDetector(TextDetectionOptions<T> options);

    /// <summary>Confidence threshold used when the test does not vary it.</summary>
    protected virtual double DetectConfidenceThreshold => 0.05;

    private static List<(double X, double Y)> PolygonOf(TextRegion<T> region)
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

    /// <summary>
    /// Exercises actual neural forwards and decoders using live trainable heads with analytically
    /// known outputs. This is a numerical pipeline fixture, not a trained text-recognition claim.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task Detect_ControlledPositiveHead_ShouldProduceScorableRegions()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        using var detector = CreatePositiveTextDetector(new TextDetectionOptions<T>
        {
            InputSize = new[] { 64, 64 },
            Size = ModelSize.Nano
        });
        detector.SetTrainingMode(false);
        var image = new Tensor<T>(new[] { 1, 3, 64, 64 });
        // A deterministic "HI" bitmap avoids font, platform, and image-file dependencies.
        // Controlled heads below make this a decoder contract, not a recognition-accuracy claim.
        byte[][] glyphs =
        {
            new byte[] { 0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001 },
            new byte[] { 0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111 }
        };
        for (int glyph = 0; glyph < glyphs.Length; glyph++)
            for (int row = 0; row < 7; row++)
                for (int column = 0; column < 5; column++)
                    if ((glyphs[glyph][row] & (1 << (4 - column))) != 0)
                        for (int dy = 0; dy < 4; dy++)
                            for (int dx = 0; dx < 4; dx++)
                                for (int channel = 0; channel < 3; channel++)
                                    image[0, channel, 18 + 4 * row + dy, 8 + 24 * glyph + 4 * column + dx] = ToT(1);

        detector.Predict(image); // Resolve actual lazy layer shapes before accessing live weights.
        var trainable = detector.GetParameterStateChunks()
            .Where(chunk => chunk.Role == ParameterSlotRole.Trainable).ToArray();
        Assert.NotEmpty(trainable);
        Assert.All(trainable, chunk => Assert.True(chunk.IsWritableInPlace, chunk.StableId));
        foreach (var chunk in trainable) chunk.Tensor.Fill(ToT(0));

        // These typed contracts describe distinct real decoders. Unknown future architectures must
        // add an explicit positive oracle; they must not silently inherit an empty-output pass.
        switch (detector)
        {
            case CRAFT<T>:
                // One 32x32 component: niter = int(sqrt(1024 * 32 / 1024) * 2) = 11, so the reference dilation
                // grows each side by (1 + 11) / 2 = 6 cells, [-6, 37] at stride 2, clipped to the 64x64 image.
                AssertPositiveTextResult(detector.Detect(image, DetectConfidenceThreshold), 1, (0, 0, 64, 64));
                break;
            case DBNet<T>:
                // The uniform head marks the whole 16x16 map as one kernel: its 15x15 minimum-area rectangle is
                // dilated by D' = A' r' / L' = 225 * 1.5 / 60 = 5.625 cells, i.e. [-22.5, 82.5] px at stride 4,
                // and clipped to the 64x64 image. Before the unclip existed this pinned the kernel, (0,0)-(63,63).
                AssertPositiveTextResult(detector.Detect(image, DetectConfidenceThreshold), 1, (0, 0, 64, 64));
                break;
            case EAST<T>:
                // The actual RBOX head is the unique trainable five-element bias. A changed or
                // ambiguous layout fails here instead of guessing which equal-shaped tensor to edit.
                var geometryBias = Assert.Single(trainable,
                    chunk => chunk.Tensor.Rank == 1 && chunk.Tensor.Length == 5).Tensor;
                // EAST bounds its RBOX head as the paper does: each distance is sigmoid(raw) * map side (8 here)
                // and the angle is (sigmoid(raw) - 0.5) * pi / 2. The biases are the inverse of the geometry wanted.
                double DistanceBias(double distance) => Math.Log(distance / (8.0 - distance));
                var overlappingGeometry = new[] { 2.0, 2.9, 2.0, 2.9, 0.0 };
                for (int index = 0; index < 4; index++)
                    geometryBias[index] = ToT(DistanceBias(overlappingGeometry[index]));
                geometryBias[4] = ToT(0);
                var raw = detector.Predict(image);
                // Public Predict flattens/concatenates every head per image: 64 score cells,
                // followed by five geometry channels with the same 8-by-8 row-major grid.
                Assert.Equal(new[] { 1, 6 * 8 * 8 }, raw.Shape.ToArray());
                for (int row = 0; row < 8; row++)
                    for (int column = 0; column < 8; column++)
                    {
                        int cell = row * 8 + column;
                        Assert.Equal(0.5, ToD(raw[0, cell]), 10);
                        for (int index = 0; index < overlappingGeometry.Length; index++)
                            Assert.Equal(overlappingGeometry[index], ToD(raw[0, (index + 1) * 64 + cell]), 10);
                    }
                // Locality-aware NMS, worked by hand: each cell's box is 46.4x32 px. Walking a row, the running
                // score-weighted merge absorbs cells 0-6: cell 6 is 28 px from the mean of cells 0-5, IoU
                // 18.4 / 74.4 = 0.247. Cell 7 is 32 px from the mean 28 of cells 0-6, IoU 14.4 / 78.4 = 0.184, below
                // the 0.2 threshold, so it starts its own group. (A distance of 3 put cell 7 at exactly 0.2, a tie
                // that floating-point rounding decides either way.) Standard NMS then keeps rows 0, 3 and 6 of
                // both the merged boxes and the singles: 6 regions, the first being row 0's merged box, x 28 +- 23.2,
                // y 4 +- 16. Zero distances are not a valid positive fixture.
                AssertPositiveTextResult(detector.Detect(image, DetectConfidenceThreshold), 6, (4.8, -12, 51.2, 20));

                for (int index = 0; index < 4; index++) geometryBias[index] = ToT(DistanceBias(0.25));
                geometryBias[4] = ToT(0);
                var separated = detector.Detect(image, DetectConfidenceThreshold);
                AssertPositiveTextResult(separated, 64, (2, 2, 6, 6));
                for (int index = 0; index < separated.TextRegions.Count; index++)
                {
                    double left = 8 * (index % 8) + 2;
                    double top = 8 * (index / 8) + 2;
                    AssertBoxEquals(separated.TextRegions[index], (left, top, left + 4, top + 4));
                }
                break;
            default:
                throw new InvalidOperationException("This text detector has no typed positive-fixture oracle.");
        }

        // The same initialized model and known image must reject those exact 0.5 scores when
        // the caller raises the threshold; this is not a second randomly initialized fixture.
        Assert.Empty(detector.Detect(image, 0.75).TextRegions);
    }

    /// <summary>
    /// A batch of N images must give exactly the N results of detecting each image alone.
    /// </summary>
    /// <remarks>
    /// Every text detector used to decode batch position 0 of a batched forward and return it as the
    /// whole answer, so the other images vanished without an error. The two images differ and the check
    /// requires them to give different results, so a decoder that reads item 0 for every item fails here
    /// rather than passing on two identical answers.
    /// </remarks>
    /// <summary>Training steps the overfit test may take before it must reach <see cref="OverfitHMeanFloor"/>.</summary>
    protected virtual int OverfitStepBudget => 60;

    /// <summary>The ICDAR H-mean (IoU 0.5) that the overfit test requires after training.</summary>
    protected virtual double OverfitHMeanFloor => 0.99;

    /// <summary>
    /// The paper objective has to fit: trained through <c>TrainTextDetections</c> on one image of two words, the
    /// detector must come to find exactly those words. That means ICDAR H-mean 1.0 at IoU 0.5, from a start
    /// that does not already find them. A loss with the wrong sign, targets in the wrong place or a head the
    /// update never reaches cannot pass this. A loss that only shrinks also cannot pass, because H-mean is
    /// measured on the decoded detections at the model's configured threshold, not on the loss.
    /// </summary>
    [Fact(Timeout = 300000)]
    public async Task TrainTextDetections_OverfitsOneImage_ToFullHMean()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        using var detector = CreatePositiveTextDetector(new TextDetectionOptions<T>
        {
            InputSize = new[] { 64, 64 },
            Size = ModelSize.Nano
        });
        var image = new Tensor<T>(new[] { 1, 3, 64, 64 });
        var words = new[]
        {
            SyntheticTextImages.Draw(image, 0, "HI", 6, 6, 3),
            SyntheticTextImages.Draw(image, 0, "TEL", 8, 38, 3),
        };
        var batch = new TextDetectionTrainingBatch(new[]
        {
            words.Select((w, i) => TextPolygonTarget.FromBox(w.Left, w.Top, w.Right, w.Bottom, i == 0 ? "HI" : "TEL"))
        });
        var groundTruth = words.Select(w => TextRegion<T>.FromPolygon(new List<(T X, T Y)>
        {
            (ToT(w.Left), ToT(w.Top)), (ToT(w.Right), ToT(w.Top)), (ToT(w.Right), ToT(w.Bottom)), (ToT(w.Left), ToT(w.Bottom)),
        }, ToT(1))).ToList();
        var metrics = new TextDetectionMetrics<T>();
        double HMean()
        {
            detector.SetTrainingMode(false);
            // update_bn: eval-mode statistics for the current weights, not the momentum average of earlier ones.
            BatchNormRecalibration.Recalibrate<T>(detector, () => detector.Predict(image));
            var found = detector.Detect(image).TextRegions;
            return metrics.Evaluate(new[] { found }, new[] { groundTruth }).HMean;
        }

        double initial = HMean();
        var trajectory = new List<string> { $"0: hmean {initial:F3}" };
        double final = initial;
        for (int step = 1; step <= OverfitStepBudget && final < OverfitHMeanFloor; step++)
        {
            detector.TrainTextDetections(image, batch);
            double loss = ToD(detector.GetLastLoss());
            Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"Step {step} reported a non-finite loss {loss}.");
            if (step % 5 == 0 || step == OverfitStepBudget)
            {
                final = HMean();
                trajectory.Add($"{step}: loss {loss:G4} hmean {final:F3}");
            }
        }

        Assert.True(initial < OverfitHMeanFloor,
            $"The untrained detector already scores H-mean {initial:F3}, so this fixture cannot show learning.");
        Assert.True(final >= OverfitHMeanFloor,
            $"After {OverfitStepBudget} steps H-mean is {final:F3}, below {OverfitHMeanFloor}. " + string.Join("; ", trajectory));
    }
    [Fact(Timeout = 120000)]
    public async Task DetectBatch_EqualsDetectingEachImageAlone()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateTextDetector();
        detector.SetTrainingMode(false);

        var first = CreateRandomImage(rng);
        var second = CreateRandomImage(rng);
        var batchShape = first.Shape.ToArray();
        batchShape[0] = 2;
        var batch = new Tensor<T>(batchShape);
        for (int i = 0; i < first.Length; i++)
        {
            batch[i] = first[i];
            batch[first.Length + i] = second[i];
        }

        // Zero keeps every scored region, so the comparison sees as much decoder output as it can.
        var aloneFirst = detector.Detect(first, 0.0);
        var aloneSecond = detector.Detect(second, 0.0);
        var batched = detector.DetectBatch(batch, 0.0);

        Assert.Equal(2, batched.Count);
        AssertSameRegions(aloneFirst, batched[0], "batch item 0");
        AssertSameRegions(aloneSecond, batched[1], "batch item 1");
        Assert.True(aloneFirst.TextRegions.Count + aloneSecond.TextRegions.Count > 0,
            "Neither image produced a region, so this fixture cannot tell batch items apart.");
        Assert.NotEqual(RegionSignature(aloneFirst), RegionSignature(aloneSecond));

        // One result cannot describe two images, so Detect refuses a batch instead of dropping one.
        Assert.Throws<ArgumentException>(() => detector.Detect(batch));
    }

    private static void AssertSameRegions(TextDetectionResult<T> expected, TextDetectionResult<T> actual, string label)
    {
        Assert.True(expected.TextRegions.Count == actual.TextRegions.Count,
            $"{label}: {actual.TextRegions.Count} regions batched, {expected.TextRegions.Count} alone.");
        for (int r = 0; r < expected.TextRegions.Count; r++)
        {
            var (el, et, er, eb) = expected.TextRegions[r].Box.ToXYXY();
            var (al, at, ar, ab) = actual.TextRegions[r].Box.ToXYXY();
            // A batched forward may reduce in a different order than a single one, hence a tolerance.
            Assert.Equal(el, al, 6); Assert.Equal(et, at, 6); Assert.Equal(er, ar, 6); Assert.Equal(eb, ab, 6);
            Assert.Equal(ToD(expected.TextRegions[r].Confidence), ToD(actual.TextRegions[r].Confidence), 6);
        }
    }

    private static string RegionSignature(TextDetectionResult<T> result)
        => string.Join(";", result.TextRegions.Select(region =>
        {
            var (left, top, right, bottom) = region.Box.ToXYXY();
            return $"{left:F4},{top:F4},{right:F4},{bottom:F4},{ToD(region.Confidence):F6}";
        }));

    internal static void AssertPositiveTextResult(TextDetectionResult<T> result, int expectedCount,
        (double Left, double Top, double Right, double Bottom) expectedFirstBox)
    {
        Assert.NotNull(result);
        Assert.NotNull(result.TextRegions);
        Assert.Equal(expectedCount, result.TextRegions.Count);
        Assert.NotEmpty(result.TextRegions);
        Assert.Equal(64, result.ImageWidth);
        Assert.Equal(64, result.ImageHeight);
        foreach (var region in result.TextRegions)
        {
            Assert.NotEmpty(PolygonOf(region));
            AssertPolygonEnclosesArea(region);
            AssertBoxGeometricallyValid(region);
            AssertBoxBoundsPolygon(region);
            var (left, top, right, bottom) = region.Box.ToXYXY();
            Assert.All(new[] { left, top, right, bottom },
                coordinate => Assert.False(double.IsInfinity(coordinate), "Positive text box has an infinite coordinate."));
            Assert.InRange(ToD(region.Confidence), 0.0, 1.0);
            Assert.Equal(0.5, ToD(region.Confidence), 10);
        }
        AssertBoxEquals(result.TextRegions[0], expectedFirstBox);
    }

    private static void AssertBoxEquals(TextRegion<T> region,
        (double Left, double Top, double Right, double Bottom) expected)
    {
        var (left, top, right, bottom) = region.Box.ToXYXY();
        Assert.Equal(expected.Left, left, 10);
        Assert.Equal(expected.Top, top, 10);
        Assert.Equal(expected.Right, right, 10);
        Assert.Equal(expected.Bottom, bottom, 10);
    }

    private static void AssertPolygonEnclosesArea(TextRegion<T> region)
    {
        var polygon = PolygonOf(region);
        if (polygon.Count == 0) return; // Existing random-input invariants permit box-only regions.
        Assert.True(polygon.Count >= 4,
            $"Text polygon has only {polygon.Count} vertices; a quadrilateral is the minimum the ICDAR protocol accepts.");
        foreach (var (x, y) in polygon)
        {
            Assert.False(double.IsNaN(x) || double.IsNaN(y), "Text polygon has a NaN vertex.");
            Assert.False(double.IsInfinity(x) || double.IsInfinity(y), "Text polygon has an infinite vertex.");
        }
        Assert.True(TextDetectionMetrics<T>.PolygonArea(polygon) > 0.0,
            "Text polygon encloses zero area, so every IoU against it is zero and the region can never be scored as a match.");
    }

    internal static void AssertBoxGeometricallyValid(TextRegion<T> region)
    {
        Assert.NotNull(region.Box);
        var (xMin, yMin, xMax, yMax) = region.Box.ToXYXY();
        foreach (double coordinate in new[] { xMin, yMin, xMax, yMax })
        {
            Assert.False(double.IsNaN(coordinate) || double.IsInfinity(coordinate),
                "Text region box has a non-finite coordinate.");
        }
        Assert.True(xMax > xMin, $"Text region box has inverted or zero width: {xMin} to {xMax}.");
        Assert.True(yMax > yMin, $"Text region box has inverted or zero height: {yMin} to {yMax}.");
    }

    private static void AssertBoxBoundsPolygon(TextRegion<T> region)
    {
        var polygon = PolygonOf(region);
        if (polygon.Count == 0) return;
        var (xMin, yMin, xMax, yMax) = region.Box.ToXYXY();
        foreach (var (x, y) in polygon)
        {
            Assert.InRange(x, xMin - 1e-6, xMax + 1e-6);
            Assert.InRange(y, yMin - 1e-6, yMax + 1e-6);
        }
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
            AssertPolygonEnclosesArea(region);
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
            AssertBoxGeometricallyValid(region);
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
            // Consumers that cannot handle polygons fall back to the box. If the box does not
            // contain the polygon, that fallback silently crops the detected word.
            AssertBoxBoundsPolygon(region);
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
