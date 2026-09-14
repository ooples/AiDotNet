using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;
using AiDotNet.Models.Options;
using AiDotNet.Models.Parameters;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Shared numerical pipeline fixture. Every result comes from the actual model forward with live
/// trainable weights, not a substituted forward or hand-built detection list. Controlled heads
/// establish decoder contracts; they do not claim learned object-recognition accuracy.
/// </summary>
internal static class ObjectDetectionPositiveFixture<T> where T : struct
{
    internal readonly struct ExpectedDetection
    {
        public ExpectedDetection(double score, double left, double top, double right, double bottom)
            => (Score, Left, Top, Right, Bottom) = (score, left, top, right, bottom);
        public double Score { get; }
        public double Left { get; }
        public double Top { get; }
        public double Right { get; }
        public double Bottom { get; }
    }

    private enum HeadProfile { Yolo, Detr, RtDetr, Dino, FasterRcnn, CascadeRcnn }

    internal static ObjectDetectionOptions<T> CreateOptions() => new()
    {
        InputSize = new[] { 64, 64 }, Size = ModelSize.Nano, NumClasses = 2
    };

    internal static void Verify(ObjectDetectorBase<T> detector)
    {
        var profile = ProfileOf(detector);
        VerifyOne(detector, profile, suppressionDisabled: detector is YOLOv10<T>);
        if (detector is YOLOv10<T>)
        {
            // YOLOv10 intentionally defaults to NMS-free. Test that public default separately
            // from its explicitly supported NMS mode; neither is a per-model scaffold edit.
            using var withSuppression = new YOLOv10<T>(CreateOptions(), useNmsFree: false);
            Assert.Equal(1.0, detector.EffectiveNmsThreshold(0.45));
            Assert.Equal(0.45, withSuppression.EffectiveNmsThreshold(0.45));
            VerifyOne(withSuppression, HeadProfile.Yolo, suppressionDisabled: false);
        }
    }

    private static HeadProfile ProfileOf(ObjectDetectorBase<T> detector) => detector switch
    {
        YOLOv8<T> or YOLOv9<T> or YOLOv10<T> or YOLOv11<T> => HeadProfile.Yolo,
        DETR<T> => HeadProfile.Detr,
        RTDETR<T> => HeadProfile.RtDetr,
        DINO<T> => HeadProfile.Dino,
        FasterRCNN<T> => HeadProfile.FasterRcnn,
        CascadeRCNN<T> => HeadProfile.CascadeRcnn,
        _ => throw new InvalidOperationException("This object detector needs an explicit typed positive-fixture oracle.")
    };

    private static void VerifyOne(ObjectDetectorBase<T> detector, HeadProfile profile, bool suppressionDisabled)
    {
        detector.SetTrainingMode(false);
        Assert.Equal(2, detector.NumClasses);
        Assert.Equal(300, detector.MaxDetections); // No candidate is hidden by the cap in these profiles.
        var image = new Tensor<T>(new[] { 1, 3, 64, 64 });
        var normalized = new Tensor<T>(new[] { 1, 3, 64, 64 });
        for (int channel = 0; channel < 3; channel++)
            for (int y = 16; y < 48; y++)
                for (int x = 12; x < 52; x++)
                {
                    image[0, channel, y, x] = ToT(255);
                    normalized[0, channel, y, x] = ToT(1);
                }
        // Predict expects network input, whereas Detect normalizes [0,255] pixels. The two
        // tensors above represent exactly the same image at those respective public boundaries.
        detector.Predict(normalized);
        var trainable = detector.GetParameterStateChunks()
            .Where(chunk => chunk.Role == ParameterSlotRole.Trainable).ToArray();
        Assert.NotEmpty(trainable);
        Assert.All(trainable, chunk => Assert.True(chunk.IsWritableInPlace, chunk.StableId));
        foreach (var chunk in trainable) chunk.Tensor.Fill(ToT(0));
        ConfigureHead(trainable, profile);

        var raw = detector.Predict(normalized);
        var expected = ExpectedCandidates(raw, profile);
        AssertCandidatePreconditions(expected);
        AssertMatches(detector.Detect(image, 0.05, 1.0), expected);

        var suppressed = SuppressIndependently(expected, 0.45);
        Assert.Equal(profile is HeadProfile.FasterRcnn or HeadProfile.CascadeRcnn ? 39 : 1, suppressed.Count);
        Assert.True(suppressed.Count < expected.Count, "The controlled candidates must actually exercise NMS.");
        AssertMatches(detector.Detect(image, 0.05, 0.45), suppressionDisabled ? expected : suppressed);

        // Same live model and image: the threshold must reject these known scores, not merely
        // happen to produce fewer detections on another random initialization.
        Assert.All(expected, candidate => Assert.True(candidate.Score < 0.99));
        Assert.Empty(detector.Detect(image, 0.99, 0.45).Detections);
    }

    private static void ConfigureHead(ParameterChunk<T>[] trainable, HeadProfile profile)
    {
        if (profile == HeadProfile.Yolo)
        {
            var biases = trainable.Where(chunk => chunk.Tensor.Rank == 1 && chunk.Tensor.Length == 2).ToArray();
            Assert.Equal(3, biases.Length);
            double[] odds = { 1, 3, 7 };
            for (int level = 0; level < biases.Length; level++)
            {
                biases[level].Tensor[0] = ToT(Math.Log(odds[level]));
                biases[level].Tensor[1] = ToT(-10);
            }
            return;
        }
        if (profile is HeadProfile.Detr or HeadProfile.RtDetr or HeadProfile.Dino)
        {
            const int hidden = 128;
            int queries = profile == HeadProfile.Detr ? 50 : 100;
            foreach (var chunk in trainable.Where(chunk => chunk.Tensor.Rank == 1 && chunk.Tensor.Length == hidden))
                chunk.Tensor.Fill(ToT(1));
            var embeddings = trainable.Where(chunk => HasMatrixShape(chunk.Tensor, queries, hidden)).ToArray();
            Assert.Equal(profile == HeadProfile.Dino ? 2 : 1, embeddings.Length);
            foreach (var embedding in embeddings)
            {
                // Two different zero-mean query directions survive the actual normalization
                // layers. Class logits differ; all-zero tied queries cannot prove ordering.
                embedding.Tensor[0, 0] = ToT(1);
                embedding.Tensor[0, 1] = ToT(1);
                embedding.Tensor[0, 2] = ToT(-1);
                embedding.Tensor[0, 3] = ToT(-1);
                embedding.Tensor[1, 0] = ToT(1);
                embedding.Tensor[1, 1] = ToT(-1);
            }
            var weights = Assert.Single(trainable, chunk => HasMatrixShape(chunk.Tensor, hidden, 3)).Tensor;
            weights[0, 0] = ToT(1); // Actual Dense storage is [input,output], not [output,input].
            var bias = Assert.Single(trainable, chunk => chunk.Tensor.Rank == 1 && chunk.Tensor.Length == 3).Tensor;
            bias[0] = ToT(-4);
            bias[1] = ToT(-20);
            bias[2] = ToT(2); // DETR background is the last class.
            return;
        }

        int stageCount = profile == HeadProfile.CascadeRcnn ? 3 : 1;
        int headInput = profile == HeadProfile.CascadeRcnn ? 128 : 256 * 5 * 5;
        Assert.Equal(stageCount, trainable.Count(chunk => HasMatrixShape(chunk.Tensor, headInput, 3)));
        Assert.Equal(stageCount, trainable.Count(chunk => chunk.Tensor.Rank == 1 && chunk.Tensor.Length == 3));
        foreach (var chunk in trainable)
        {
            var tensor = chunk.Tensor;
            // A single real spatial channel passes through backbone, FPN, ROIAlign and the
            // classifier. Positive affine offsets keep ReLU paths live. RPN score/delta heads
            // and every ROI box-regression head remain zero, giving fixed anchor proposals.
            if (tensor.Rank == 1 && tensor.Length >= 64)
                tensor.Fill(ToT(1));
            else if (tensor.Rank == 4 && tensor.Shape[0] >= 64)
                tensor[0, 0, tensor.Shape[2] / 2, tensor.Shape[3] / 2] = ToT(1);
            else if (tensor.Rank == 2 && tensor.Shape[0] >= 64 && tensor.Shape[1] >= 64)
                tensor[0, 0] = ToT(1);
            else if (HasMatrixShape(tensor, headInput, 3))
                tensor[0, 1] = ToT(0.00001);
            else if (tensor.Rank == 1 && tensor.Length == 3)
                tensor[2] = ToT(-20); // R-CNN background is the first class, unlike DETR.
        }
    }

    private static List<ExpectedDetection> ExpectedCandidates(Tensor<T> raw, HeadProfile profile)
        => profile switch
        {
            HeadProfile.Yolo => ExpectedYolo(raw),
            HeadProfile.Detr or HeadProfile.RtDetr or HeadProfile.Dino => ExpectedDetr(raw, profile),
            HeadProfile.FasterRcnn or HeadProfile.CascadeRcnn => ExpectedRcnn(raw, profile),
            _ => throw new ArgumentOutOfRangeException(nameof(profile))
        };

    private static List<ExpectedDetection> ExpectedYolo(Tensor<T> raw)
    {
        int[] cells = { 64, 16, 4 };
        double[] odds = { 1, 3, 7 };
        Assert.Equal(new[] { 1, (2 + 4 * 16) * 84 }, raw.Shape.ToArray());
        int offset = 0;
        var expected = new List<ExpectedDetection>();
        for (int level = 0; level < cells.Length; level++)
        {
            for (int cell = 0; cell < cells[level]; cell++)
            {
                AssertClose(Math.Log(odds[level]), ToD(raw[0, offset + cell]));
                AssertClose(-10, ToD(raw[0, offset + cells[level] + cell]));
                // Zero 16-bin DFL logits give mean distance 7.5. At strides 8,16,32 every
                // cell's box covers the whole 64x64 image after the actual decoder clips it.
                expected.Add(new ExpectedDetection(odds[level] / (1 + odds[level]), 0, 0, 64, 64));
            }
            offset += 2 * cells[level];
        }
        for (int index = offset; index < raw.Length; index++) AssertClose(0, ToD(raw[0, index]));
        return expected.OrderByDescending(candidate => candidate.Score).ToList();
    }

    private static List<ExpectedDetection> ExpectedDetr(Tensor<T> raw, HeadProfile profile)
    {
        int queries = profile == HeadProfile.Detr ? 50 : 100;
        Assert.Equal(new[] { 1, queries * 7 }, raw.Shape.ToArray());
        var expected = new List<ExpectedDetection>();
        for (int query = 0; query < queries; query++)
        {
            double classLogit = NormalizedQueryFirstCoordinate(query, profile == HeadProfile.Dino ? 2 : 1) - 4;
            AssertClose(classLogit, ToD(raw[0, query * 3]));
            AssertClose(-20, ToD(raw[0, query * 3 + 1]));
            AssertClose(2, ToD(raw[0, query * 3 + 2]));
            // The decoder's public score storage is float. Include the actual background
            // probability, rather than treating the class logit as a sigmoid.
            double score = (float)(1 / (1 + Math.Exp(-20 - classLogit) + Math.Exp(2 - classLogit)));
            if (query < 2)
            {
                Assert.True(score > 0.05);
                expected.Add(new ExpectedDetection(score, 16, 16, 48, 48));
            }
            else Assert.True(score < 0.05);
        }
        for (int index = queries * 3; index < raw.Length; index++) AssertClose(0, ToD(raw[0, index]));
        return expected.OrderByDescending(candidate => candidate.Score).ToList();
    }

    private static double NormalizedQueryFirstCoordinate(int query, int embeddingCount)
    {
        var values = new double[128];
        if (query == 0)
            (values[0], values[1], values[2], values[3]) = (embeddingCount, embeddingCount, -embeddingCount, -embeddingCount);
        else if (query == 1)
            (values[0], values[1]) = (embeddingCount, -embeddingCount);
        // Nano has three decoder layers, each with three real residual + affine layer norms.
        // Projection weights are zero and projection bias, gamma, beta are one.
        for (int normalization = 0; normalization < 9; normalization++)
        {
            for (int index = 0; index < values.Length; index++) values[index] += 1;
            double mean = values.Average();
            double variance = values.Sum(value => (value - mean) * (value - mean)) / values.Length;
            double scale = Math.Sqrt(variance + 1e-6);
            for (int index = 0; index < values.Length; index++) values[index] = (values[index] - mean) / scale + 1;
        }
        return values[0];
    }

    private static List<ExpectedDetection> ExpectedRcnn(Tensor<T> raw, HeadProfile profile)
    {
        // Five real RPN grids contain 3*(16^2+8^2+4^2+2^2+1)=1023 anchors.
        // Zero RPN logits/deltas and per-level NMS leave 275 fixed proposals. Public Predict
        // concatenates final [N,3] logits, [N,12] deltas, [N,4] proposals, optional earlier
        // cascade heads, then 1023*2 objectness and 1023*4 RPN regression values.
        const int proposals = 275;
        int stageCount = profile == HeadProfile.CascadeRcnn ? 3 : 1;
        Assert.Equal(new[] { 1, proposals * (4 + stageCount * 15) + 1023 * 6 }, raw.Shape.ToArray());
        var expected = new List<ExpectedDetection>();
        for (int proposal = 0; proposal < proposals; proposal++)
        {
            AssertClose(0, ToD(raw[0, proposal * 3]));
            double foregroundLogit = ToD(raw[0, proposal * 3 + 1]);
            Assert.InRange(foregroundLogit, 3, 5);
            AssertClose(-20, ToD(raw[0, proposal * 3 + 2]));
            double score = 1 / (1 + Math.Exp(-foregroundLogit) + Math.Exp(-20 - foregroundLogit));
            int box = proposals * 15 + proposal * 4;
            expected.Add(new ExpectedDetection(score, ToD(raw[0, box]), ToD(raw[0, box + 1]),
                ToD(raw[0, box + 2]), ToD(raw[0, box + 3])));
        }
        for (int index = proposals * 3; index < proposals * 15; index++) AssertClose(0, ToD(raw[0, index]));
        for (int stage = 0; stage < stageCount - 1; stage++)
        {
            int offset = proposals * (19 + stage * 15);
            for (int proposal = 0; proposal < proposals; proposal++)
            {
                AssertClose(0, ToD(raw[0, offset + proposal * 3]));
                Assert.InRange(ToD(raw[0, offset + proposal * 3 + 1]), 3, 5);
                AssertClose(-20, ToD(raw[0, offset + proposal * 3 + 2]));
            }
            for (int index = offset + proposals * 3; index < offset + proposals * 15; index++)
                AssertClose(0, ToD(raw[0, index]));
        }
        for (int index = raw.Length - 1023 * 6; index < raw.Length; index++) AssertClose(0, ToD(raw[0, index]));
        var sorted = expected.OrderByDescending(candidate => candidate.Score).ToList();
        // Highest-score actual proposal is the stride-4, ratio-1/2 anchor centered at (62,46),
        // with width 32*sqrt(2), height 32/sqrt(2), and right edge clipped to the image.
        AssertClose(62 - 16 * Math.Sqrt(2), sorted[0].Left);
        AssertClose(46 - 16 / Math.Sqrt(2), sorted[0].Top);
        AssertClose(64, sorted[0].Right);
        AssertClose(46 + 16 / Math.Sqrt(2), sorted[0].Bottom);
        return sorted;
    }

    internal static void AssertCandidatePreconditions(IReadOnlyList<ExpectedDetection> expected)
    {
        Assert.True(expected.Count >= 2, "Ordering and NMS require at least two eligible same-class candidates.");
        Assert.True(expected.Max(candidate => candidate.Score) - expected.Min(candidate => candidate.Score) > 1e-5,
            "All-tied scores cannot prove confidence ordering.");
        Assert.True(expected.Where((candidate, index) => expected.Skip(index + 1)
            .Any(other => IntersectionOverUnion(candidate, other) > 0.45)).Any(),
            "At least one same-class overlap must require actual suppression.");
    }

    internal static List<ExpectedDetection> SuppressIndependently(IReadOnlyList<ExpectedDetection> ordered, double threshold)
    {
        // Independent greedy reference: no production NMS/BoundingBox.IoU/decoder calls.
        var kept = new List<ExpectedDetection>();
        foreach (var candidate in ordered)
            if (kept.All(winner => IntersectionOverUnion(winner, candidate) <= threshold)) kept.Add(candidate);
        return kept;
    }

    internal static void AssertMatches(DetectionResult<T> actual, IReadOnlyList<ExpectedDetection> expected)
    {
        Assert.NotNull(actual);
        Assert.NotNull(actual.Detections);
        Assert.NotEmpty(expected);
        Assert.NotEmpty(actual.Detections);
        Assert.Equal(expected.Count, actual.Detections.Count);
        Assert.Equal(64, actual.ImageWidth);
        Assert.Equal(64, actual.ImageHeight);
        for (int index = 0; index < expected.Count; index++)
        {
            var detection = actual.Detections[index];
            Assert.NotNull(detection.Box);
            Assert.Equal(0, detection.ClassId);
            double score = ToD(detection.Confidence);
            Assert.InRange(score, 0.05, 1.0);
            AssertClose(expected[index].Score, score);
            if (index > 0) Assert.True(ToD(actual.Detections[index - 1].Confidence) >= score);
            var (left, top, right, bottom) = detection.Box.ToXYXY();
            foreach (double coordinate in new[] { left, top, right, bottom }) Assert.InRange(coordinate, 0, 64);
            Assert.True(right > left && bottom > top, "Positive detections must enclose nonzero area.");
            AssertClose(expected[index].Left, left);
            AssertClose(expected[index].Top, top);
            AssertClose(expected[index].Right, right);
            AssertClose(expected[index].Bottom, bottom);
        }
    }

    private static double IntersectionOverUnion(ExpectedDetection a, ExpectedDetection b)
    {
        double intersection = Math.Max(0, Math.Min(a.Right, b.Right) - Math.Max(a.Left, b.Left))
            * Math.Max(0, Math.Min(a.Bottom, b.Bottom) - Math.Max(a.Top, b.Top));
        double union = (a.Right - a.Left) * (a.Bottom - a.Top) + (b.Right - b.Left) * (b.Bottom - b.Top) - intersection;
        return intersection / union;
    }

    private static bool HasMatrixShape(Tensor<T> tensor, int rows, int columns)
        => tensor.Rank == 2 && tensor.Shape[0] == rows && tensor.Shape[1] == columns;
    private static void AssertClose(double expected, double actual) => Assert.InRange(Math.Abs(expected - actual), 0, 1e-6);
    private static T ToT(double value) => AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().FromDouble(value);
    private static double ToD(T value) => AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>().ToDouble(value);
}
