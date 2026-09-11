using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for object detectors (YOLOv8/9/10/11, DETR, DINO, RT-DETR, Faster R-CNN,
/// Cascade R-CNN).
/// </summary>
/// <remarks>
/// <para>
/// Adds the invariants that define a well-formed detection set, as the COCO and Pascal VOC
/// evaluation protocols assume them. Evaluation silently produces nonsense if any is violated:
/// a box with inverted corners has negative area so every IoU against it is wrong; unsorted
/// scores break the precision-recall ranking; a class id outside the label set indexes past the
/// end of the class array; and detections that survive NMS while overlapping above the threshold
/// inflate recall with duplicates of the same object.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type the detector is expressed in.</typeparam>
public abstract class ObjectDetectionTestBase<T> : DetectionModelTestBase<T>
{
    /// <summary>
    /// The model under test as a detector. Family resolution guarantees the cast: only types
    /// deriving from <c>ObjectDetectorBase</c> are assigned this test family.
    /// </summary>
    protected ObjectDetectorBase<T> CreateDetector() => (ObjectDetectorBase<T>)CreateModel();

    /// <summary>Confidence threshold used when the test does not vary it.</summary>
    protected virtual double DetectConfidenceThreshold => 0.05;

    /// <summary>NMS IoU threshold used when the test does not vary it.</summary>
    protected virtual double DetectNmsThreshold => 0.45;

    private Tensor<T> CreateBatch(Random rng, int batchSize)
    {
        var shape = (int[])InputShape.Clone();
        shape[0] = batchSize;
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = ToT(rng.NextDouble());
        }

        return tensor;
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ShouldProduceGeometricallyValidBoxes()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        var result = detector.Detect(CreateRandomImage(rng), DetectConfidenceThreshold, DetectNmsThreshold);

        Assert.NotNull(result);
        Assert.NotNull(result.Detections);
        foreach (var detection in result.Detections)
        {
            Assert.NotNull(detection.Box);
            var (xMin, yMin, xMax, yMax) = detection.Box.ToXYXY();

            Assert.False(double.IsNaN(xMin) || double.IsNaN(yMin) || double.IsNaN(xMax) || double.IsNaN(yMax),
                "Detection box has a NaN coordinate.");
            Assert.False(double.IsInfinity(xMin) || double.IsInfinity(yMin)
                || double.IsInfinity(xMax) || double.IsInfinity(yMax),
                "Detection box has an infinite coordinate.");
            Assert.True(xMax > xMin,
                $"Detection box has inverted or zero width: x1={xMin}, x2={xMax}.");
            Assert.True(yMax > yMin,
                $"Detection box has inverted or zero height: y1={yMin}, y2={yMax}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ScoresShouldBeInUnitRange()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        var result = detector.Detect(CreateRandomImage(rng), DetectConfidenceThreshold, DetectNmsThreshold);

        foreach (var detection in result.Detections)
        {
            double confidence = ToD(detection.Confidence);
            Assert.InRange(confidence, 0.0, 1.0);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ScoresShouldBeDescending()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        var detections = detector
            .Detect(CreateRandomImage(rng), DetectConfidenceThreshold, DetectNmsThreshold)
            .Detections;

        // COCO evaluation ranks by confidence; emitting them already ranked is the convention
        // every consumer (and the MaxDetections cap, which truncates the tail) relies on.
        for (int i = 1; i < detections.Count; i++)
        {
            Assert.True(
                ToD(detections[i - 1].Confidence) >= ToD(detections[i].Confidence) - 1e-9,
                $"Detections are not in descending confidence order at index {i}: "
                + $"{ToD(detections[i - 1].Confidence)} then {ToD(detections[i].Confidence)}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ScoresShouldClearTheRequestedThreshold()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        const double threshold = 0.3;
        var detections = detector
            .Detect(CreateRandomImage(rng), threshold, DetectNmsThreshold)
            .Detections;

        foreach (var detection in detections)
        {
            Assert.True(
                ToD(detection.Confidence) >= threshold - 1e-9,
                $"Detection kept with confidence {ToD(detection.Confidence)} below the "
                + $"requested threshold {threshold}.");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ClassIdsShouldIndexTheLabelSet()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        var detections = detector
            .Detect(CreateRandomImage(rng), DetectConfidenceThreshold, DetectNmsThreshold)
            .Detections;

        int classCount = detector.NumClasses;
        foreach (var detection in detections)
        {
            Assert.InRange(detection.ClassId, 0, classCount - 1);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ShouldRespectMaxDetections()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        // A very low threshold is the case that exercises the cap: without it the raw head can
        // emit thousands of boxes.
        var detections = detector.Detect(CreateRandomImage(rng), 0.0, DetectNmsThreshold).Detections;

        Assert.True(
            detections.Count <= detector.MaxDetections,
            $"Detector returned {detections.Count} detections, above its MaxDetections "
            + $"of {detector.MaxDetections}.");
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_RaisingTheConfidenceThresholdCannotAddDetections()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();
        var image = CreateRandomImage(rng);

        int lenient = detector.Detect(image, 0.05, DetectNmsThreshold).Detections.Count;
        int strict = detector.Detect(image, 0.9, DetectNmsThreshold).Detections.Count;

        // Monotonicity is what makes a precision-recall curve well defined: sweeping the
        // threshold upward must only ever remove detections.
        Assert.True(
            strict <= lenient,
            $"Raising the confidence threshold increased the detection count: {lenient} at 0.05, "
            + $"{strict} at 0.9.");
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_SurvivorsShouldNotOverlapAboveTheNmsThreshold()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        const double nmsThreshold = 0.45;
        var detections = detector.Detect(CreateRandomImage(rng), 0.05, nmsThreshold).Detections;

        // Per-class NMS is the standard; a class-agnostic implementation also satisfies this,
        // so the weaker per-class claim is the right one to assert.
        for (int i = 0; i < detections.Count; i++)
        {
            for (int j = i + 1; j < detections.Count; j++)
            {
                if (detections[i].ClassId != detections[j].ClassId)
                {
                    continue;
                }

                double iou = detections[i].Box.IoU(detections[j].Box);
                Assert.True(
                    iou <= nmsThreshold + 1e-9,
                    $"Two surviving class-{detections[i].ClassId} boxes overlap at IoU {iou}, "
                    + $"above the NMS threshold {nmsThreshold}. Non-maximum suppression did not "
                    + "remove the duplicate.");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ShouldReportTheSourceImageDimensions()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        var result = detector.Detect(CreateRandomImage(rng), DetectConfidenceThreshold, DetectNmsThreshold);

        // Boxes are only interpretable against the frame they were measured in, so the result
        // has to carry non-zero dimensions.
        Assert.True(result.ImageWidth > 0, "DetectionResult.ImageWidth was not populated.");
        Assert.True(result.ImageHeight > 0, "DetectionResult.ImageHeight was not populated.");
    }

    [Fact(Timeout = 120000)]
    public async Task Detect_ShouldBeDeterministic()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();
        var image = CreateRandomImage(rng);

        var first = detector.Detect(image, DetectConfidenceThreshold, DetectNmsThreshold).Detections;
        var second = detector.Detect(image, DetectConfidenceThreshold, DetectNmsThreshold).Detections;

        Assert.Equal(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
        {
            Assert.Equal(first[i].ClassId, second[i].ClassId);
            Assert.Equal(ToD(first[i].Confidence), ToD(second[i].Confidence), 10);
            Assert.Equal(0.0, 1.0 - first[i].Box.IoU(second[i].Box), 8);
        }
    }

    [Fact(Timeout = 180000)]
    public async Task DetectBatch_ShouldAgreeWithPerImageDetect()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        var batch = CreateBatch(rng, 2);
        var batched = detector.DetectBatch(batch, DetectConfidenceThreshold, DetectNmsThreshold);

        Assert.NotNull(batched);
        Assert.NotNull(batched.Results);
        Assert.Equal(2, batched.Results.Count);

        // Batching is a throughput optimisation and must not change what is detected. A batch
        // path that indexes the wrong slice of the output would show up here and nowhere else.
        for (int i = 0; i < 2; i++)
        {
            var single = detector
                .Detect(ExtractImage(batch, i), DetectConfidenceThreshold, DetectNmsThreshold)
                .Detections;
            var fromBatch = batched.Results[i].Detections;

            Assert.Equal(single.Count, fromBatch.Count);
            for (int d = 0; d < single.Count; d++)
            {
                Assert.Equal(single[d].ClassId, fromBatch[d].ClassId);
                Assert.Equal(ToD(single[d].Confidence), ToD(fromBatch[d].Confidence), 8);
            }
        }
    }

    private Tensor<T> ExtractImage(Tensor<T> batch, int index)
    {
        var shape = (int[])batch.Shape.Clone();
        shape[0] = 1;

        int stride = 1;
        for (int d = 1; d < batch.Shape.Length; d++)
        {
            stride *= batch.Shape[d];
        }

        var image = new Tensor<T>(shape);
        for (int i = 0; i < stride; i++)
        {
            image[i] = batch[(index * stride) + i];
        }

        return image;
    }
}

/// <summary>Default-precision alias used by the generated fixtures.</summary>
public abstract class ObjectDetectionTestBase : ObjectDetectionTestBase<double> { }
