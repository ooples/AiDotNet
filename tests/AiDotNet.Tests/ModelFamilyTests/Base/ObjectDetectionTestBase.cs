using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Models.Options;
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
    where T : struct
{
    /// <summary>
    /// The model under test as a detector. Family resolution guarantees the cast: only types
    /// deriving from <c>ObjectDetectorBase</c> are assigned this test family.
    /// </summary>
    protected ObjectDetectorBase<T> CreateDetector() => (ObjectDetectorBase<T>)CreateModel();

    /// <summary>Generated options factory for a bounded positive fixture; normal defaults stay unchanged.</summary>
    protected abstract ObjectDetectorBase<T> CreatePositiveObjectDetector(ObjectDetectionOptions<T> options);

    /// <summary>Used only by generated models that implement the real typed training capability.</summary>
    protected void VerifySemanticDetectionTraining()
    {
        foreach (bool emptyTargets in new[] { false, true })
        {
            using var detector = CreatePositiveObjectDetector(ObjectDetectionPositiveFixture<T>.CreateOptions());
            switch (detector)
            {
                case AiDotNet.ComputerVision.Detection.ObjectDetection.DETR.DETR<T>:
                    VerifyDetrSemanticStep(detector, emptyTargets);
                    break;
                case AiDotNet.ComputerVision.Detection.ObjectDetection.DETR.DINO<T>:
                case AiDotNet.ComputerVision.Detection.ObjectDetection.DETR.RTDETR<T>:
                    VerifySigmoidSetSemanticStep(detector, emptyTargets);
                    break;
                case AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO.YOLOv8<T>:
                case AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO.YOLOv9<T>:
                case AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO.YOLOv10<T>:
                case AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO.YOLOv11<T>:
                    VerifyTaskAlignedSemanticStep(detector, emptyTargets);
                    break;
                default:
                    Assert.Fail($"{detector.GetType().Name} implements semantic detection training without an independent objective oracle.");
                    break;
            }
        }
    }

    /// <summary>
    /// Checks one DINO (focal) or RT-DETR (varifocal) step on the live, randomly initialized heads: the
    /// recorded loss must equal an independent evaluation of the published objective on the exact
    /// pre-step predictions, and both heads must move.
    /// </summary>
    internal static void VerifySigmoidSetSemanticStep(ObjectDetectorBase<T> detector, bool emptyTargets,
        Action<Tensor<T>, AiDotNet.ComputerVision.Detection.DetectionTrainingBatch<T>>? trainingStep = null)
    {
        bool varifocal = detector is AiDotNet.ComputerVision.Detection.ObjectDetection.DETR.RTDETR<T>;
        Assert.True(varifocal || detector is AiDotNet.ComputerVision.Detection.ObjectDetection.DETR.DINO<T>);
        var training = Assert.IsAssignableFrom<AiDotNet.Interfaces.IDetectionTrainingModel<T>>(detector);
        var ops = MathHelper.GetNumericOperations<T>();
        const int classes = 2; // ObjectDetectionPositiveFixture.CreateOptions
        using var input = new Tensor<T>(new[] { 1, 3, 64, 64 });
        for (int index = 0; index < input.Length; index++)
            input[index] = ops.FromDouble(((index * 37) % 101) / 101.0);

        using var before = detector.Predict(input);
        Assert.Equal(0, before.Length % (classes + 4));
        int queries = before.Length / (classes + 4);
        var logits = new double[queries * classes];
        var boxes = new double[queries * 4];
        for (int index = 0; index < logits.Length; index++) logits[index] = ops.ToDouble(before[index]);
        for (int index = 0; index < boxes.Length; index++)
            boxes[index] = 1 / (1 + Math.Exp(-ops.ToDouble(before[logits.Length + index])));

        var gold = new[] { 0.55, 0.45, 0.3, 0.35 };
        const int goldClass = 1;
        var target = new AiDotNet.ComputerVision.Detection.DetectionTrainingTarget<T>(goldClass,
            ops.FromDouble(gold[0]), ops.FromDouble(gold[1]), ops.FromDouble(gold[2]), ops.FromDouble(gold[3]));
        var batch = new AiDotNet.ComputerVision.Detection.DetectionTrainingBatch<T>(new[]
        {
            emptyTargets ? Array.Empty<AiDotNet.ComputerVision.Detection.DetectionTrainingTarget<T>>() : new[] { target }
        });
        double expected = IndependentSigmoidSetObjective(logits, boxes, queries, classes,
            emptyTargets ? -1 : goldClass, gold, varifocal);

        if (trainingStep is null) training.TrainDetections(input, batch);
        else trainingStep(input, batch);

        double actual = ops.ToDouble(detector.GetLastLoss());
        double tolerance = typeof(T) == typeof(float) ? 2e-4 * Math.Max(1, Math.Abs(expected)) : 1e-8;
        Assert.True(Math.Abs(expected - actual) <= tolerance, $"Expected loss {expected:R}; recorded {actual:R}.");

        using var after = detector.Predict(input);
        Assert.Contains(Enumerable.Range(0, logits.Length), index => !Equals(before[index], after[index]));
        if (!emptyTargets)
            Assert.Contains(Enumerable.Range(logits.Length, boxes.Length), index => !Equals(before[index], after[index]));
    }

    /// <summary>
    /// Checks one task-aligned YOLO step on the live heads: the recorded loss must equal the independent
    /// oracle on the exact pre-step head outputs (both YOLOv10 heads: top-1 and top-10), and the model moves.
    /// </summary>
    internal static void VerifyTaskAlignedSemanticStep(ObjectDetectorBase<T> detector, bool emptyTargets,
        Action<Tensor<T>, AiDotNet.ComputerVision.Detection.DetectionTrainingBatch<T>>? trainingStep = null)
    {
        var training = Assert.IsAssignableFrom<AiDotNet.Interfaces.IDetectionTrainingModel<T>>(detector);
        var ops = MathHelper.GetNumericOperations<T>();
        const int classes = 2; // ObjectDetectionPositiveFixture.CreateOptions
        const int regMax = 16;
        const int imageSize = 64;
        int[] strides = { 8, 16, 32 };
        using var input = new Tensor<T>(new[] { 1, 3, imageSize, imageSize });
        for (int index = 0; index < input.Length; index++)
            input[index] = ops.FromDouble(((index * 37) % 101) / 101.0);

        using var before = detector.Predict(input);
        var beforeValues = before.ToArray().Select(value => ops.ToDouble(value)).ToArray();
        var heads = new List<(TaskAlignedDetectionOracle.Level[] Levels, int TopK)>();
        if (detector is AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO.YOLOv10<T> yolo10)
        {
            var outputs = yolo10.ForwardTrainingHeads(input)
                .Select(output => output.ToArray().Select(value => ops.ToDouble(value)).ToArray()).ToList();
            Assert.Equal(4 * strides.Length, outputs.Count);
            heads.Add((OracleLevels(outputs.Take(2 * strides.Length).ToList()), 1));
            heads.Add((OracleLevels(outputs.Skip(2 * strides.Length).ToList()), 10));
        }
        else
        {
            var outputs = new List<double[]>();
            int offset = 0;
            foreach (int width in new[] { classes, 4 * regMax })
                foreach (int stride in strides)
                {
                    int length = width * (imageSize / stride) * (imageSize / stride);
                    outputs.Add(beforeValues.Skip(offset).Take(length).ToArray());
                    offset += length;
                }
            Assert.Equal(beforeValues.Length, offset);
            heads.Add((OracleLevels(outputs), 10));
        }

        var gold = new[] { 0.55, 0.45, 0.3, 0.35 };
        var oracleGold = emptyTargets
            ? Array.Empty<TaskAlignedDetectionOracle.Gold>()
            : new[] { new TaskAlignedDetectionOracle.Gold(1, gold[0], gold[1], gold[2], gold[3]) };
        double expected = heads.Sum(head => TaskAlignedDetectionOracle.Loss(head.Levels, 1, classes, regMax,
            TaskAlignedDetectionOracle.Assign(head.Levels, 1, classes, regMax, new[] { oracleGold }, imageSize, imageSize, head.TopK)));

        var batch = new AiDotNet.ComputerVision.Detection.DetectionTrainingBatch<T>(new[]
        {
            oracleGold.Select(g => new AiDotNet.ComputerVision.Detection.DetectionTrainingTarget<T>(g.ClassId,
                ops.FromDouble(g.CenterX), ops.FromDouble(g.CenterY), ops.FromDouble(g.Width), ops.FromDouble(g.Height))).ToArray()
        });
        if (trainingStep is null) training.TrainDetections(input, batch);
        else trainingStep(input, batch);

        double actual = ops.ToDouble(detector.GetLastLoss());
        double tolerance = typeof(T) == typeof(float) ? 2e-4 * Math.Max(1, Math.Abs(expected)) : 1e-7 * Math.Max(1, Math.Abs(expected));
        Assert.True(Math.Abs(expected - actual) <= tolerance, $"Expected loss {expected:R}; recorded {actual:R}.");

        using var after = detector.Predict(input);
        int classValues = strides.Sum(stride => classes * (imageSize / stride) * (imageSize / stride));
        Assert.Contains(Enumerable.Range(0, classValues), index => beforeValues[index] != ops.ToDouble(after[index]));
        if (!emptyTargets)
            Assert.Contains(Enumerable.Range(classValues, beforeValues.Length - classValues), index => beforeValues[index] != ops.ToDouble(after[index]));

        TaskAlignedDetectionOracle.Level[] OracleLevels(IReadOnlyList<double[]> outputs) => strides
            .Select((stride, level) => new TaskAlignedDetectionOracle.Level(outputs[level], outputs[strides.Length + level],
                imageSize / stride, imageSize / stride, stride)).ToArray();
    }

    /// <summary>
    /// DINO: sigmoid focal loss (alpha 0.25, gamma 2), matching costs 2/5/2, weights 1/5/2.
    /// RT-DETR: varifocal loss (alpha 0.75, gamma 2) toward the matched box IoU, same costs and weights.
    /// Both match with Deformable DETR's focal class cost (alpha 0.25, gamma 2).
    /// </summary>
    private static double IndependentSigmoidSetObjective(double[] logits, double[] boxes, int queries, int classes,
        int goldClass, double[] gold, bool varifocal)
    {
        static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        static double Softplus(double x) => x > 0 ? x + Math.Log(1 + Math.Exp(-x)) : Math.Log(1 + Math.Exp(x));
        double[] Box(int query) => boxes.Skip(query * 4).Take(4).ToArray();

        int matched = -1;
        if (goldClass >= 0)
        {
            double best = double.PositiveInfinity;
            for (int query = 0; query < queries; query++)
            {
                double logit = logits[query * classes + goldClass];
                double p = Sigmoid(logit);
                double classCost = 0.25 * Math.Pow(1 - p, 2) * Softplus(-logit) - 0.75 * Math.Pow(p, 2) * Softplus(logit);
                var box = Box(query);
                double l1 = box.Zip(gold, (left, right) => Math.Abs(left - right)).Sum();
                double cost = 2 * classCost + 5 * l1 - 2 * PlainGIoU(box, gold);
                if (cost < best) { best = cost; matched = query; }
            }
        }

        double quality = matched >= 0 ? PlainIoU(Box(matched), gold) : 0;
        double classification = 0;
        for (int index = 0; index < logits.Length; index++)
        {
            double x = logits[index];
            double p = Sigmoid(x);
            bool positive = matched >= 0 && index == matched * classes + goldClass;
            if (varifocal)
            {
                double q = positive ? quality : 0;
                double weight = positive ? q : 0.75 * p * p;
                classification += weight * (Softplus(x) - q * x);
            }
            else
            {
                classification += positive
                    ? 0.25 * Math.Pow(1 - p, 2) * Softplus(-x)
                    : 0.75 * p * p * Softplus(x);
            }
        }

        if (matched < 0) return classification;
        var predicted = Box(matched);
        double boxL1 = predicted.Zip(gold, (left, right) => Math.Abs(left - right)).Sum();
        const double stabilizer = 1e-7; // Engine GIoU loss contract, as in IndependentDetrBoxObjective.
        var (intersection, union, enclosure) = Overlap(predicted, gold);
        double giouLoss = 1 - intersection / (union + stabilizer) + (enclosure - union) / (enclosure + stabilizer);
        return classification + 5 * boxL1 + 2 * giouLoss;
    }

    private static (double Intersection, double Union, double Enclosure) Overlap(double[] predicted, double[] target)
    {
        var p = new[] { predicted[0] - predicted[2] / 2, predicted[1] - predicted[3] / 2,
            predicted[0] + predicted[2] / 2, predicted[1] + predicted[3] / 2 };
        var t = new[] { target[0] - target[2] / 2, target[1] - target[3] / 2,
            target[0] + target[2] / 2, target[1] + target[3] / 2 };
        double intersection = Math.Max(0, Math.Min(p[2], t[2]) - Math.Max(p[0], t[0]))
            * Math.Max(0, Math.Min(p[3], t[3]) - Math.Max(p[1], t[1]));
        double union = predicted[2] * predicted[3] + target[2] * target[3] - intersection;
        double enclosure = (Math.Max(p[2], t[2]) - Math.Min(p[0], t[0])) * (Math.Max(p[3], t[3]) - Math.Min(p[1], t[1]));
        return (intersection, union, enclosure);
    }

    private static double PlainIoU(double[] predicted, double[] target)
    {
        var (intersection, union, _) = Overlap(predicted, target);
        return union <= 0 ? 0 : intersection / union;
    }

    private static double PlainGIoU(double[] predicted, double[] target)
    {
        var (intersection, union, enclosure) = Overlap(predicted, target);
        return (union <= 0 ? 0 : intersection / union) - (enclosure <= 0 ? 0 : (enclosure - union) / enclosure);
    }

    /// <summary>Checks an exact one-step task objective on actual live DETR heads; no forward is replaced.</summary>
    internal static void VerifyDetrSemanticStep(ObjectDetectorBase<T> detector, bool emptyTargets,
        Action<Tensor<T>, AiDotNet.ComputerVision.Detection.DetectionTrainingBatch<T>>? trainingStep = null,
        double boxLogit = 0)
    {
        // Other families need their own assignment/head oracle before claiming this contract.
        Assert.IsAssignableFrom<AiDotNet.ComputerVision.Detection.ObjectDetection.DETR.DETR<T>>(detector);
        var training = Assert.IsAssignableFrom<AiDotNet.Interfaces.IDetectionTrainingModel<T>>(detector);
        var ops = MathHelper.GetNumericOperations<T>();
        using var input = new Tensor<T>(new[] { 1, 3, 64, 64 });
        using (detector.Predict(input)) { }
        var chunks = detector.GetParameterStateChunks()
            .Where(chunk => chunk.Role == AiDotNet.Models.Parameters.ParameterSlotRole.Trainable).ToArray();
        Assert.NotEmpty(chunks);
        Assert.All(chunks, chunk => Assert.True(chunk.IsWritableInPlace, chunk.StableId));
        foreach (var chunk in chunks) chunk.Tensor.Fill(ops.Zero);
        var classBias = Assert.Single(chunks, chunk => chunk.Tensor.Rank == 1 && chunk.Tensor.Length == 3).Tensor;
        var boxBias = Assert.Single(chunks, chunk => chunk.Tensor.Rank == 1 && chunk.Tensor.Length == 4).Tensor;
        boxBias.Fill(ops.FromDouble(boxLogit));
        using var before = detector.Predict(input);
        Assert.Equal(new[] { 1, 50 * 7 }, before.Shape.ToArray());
        for (int index = 0; index < 50 * 3; index++) Assert.Equal(0, ops.ToDouble(before[index]));
        for (int index = 50 * 3; index < before.Length; index++) Assert.Equal(ops.FromDouble(boxLogit), before[index]);
        double actualLogit = ops.ToDouble(ops.FromDouble(boxLogit));
        double boxCoordinate = 1 / (1 + Math.Exp(-actualLogit));

        var targetBox = new[] { 0.65, 0.57, 0.25, 0.31 };
        var target = new AiDotNet.ComputerVision.Detection.DetectionTrainingTarget<T>(0,
            ops.FromDouble(targetBox[0]), ops.FromDouble(targetBox[1]), ops.FromDouble(targetBox[2]), ops.FromDouble(targetBox[3]));
        var batch = new AiDotNet.ComputerVision.Detection.DetectionTrainingBatch<T>(new[]
        {
            emptyTargets ? Array.Empty<AiDotNet.ComputerVision.Detection.DetectionTrainingTarget<T>>() : new[] { target }
        });
        if (trainingStep is null) training.TrainDetections(input, batch);
        else trainingStep(input, batch);

        const double learningRate = 0.001;
        double denominator = emptyTargets ? 5 : 1 + 49 * 0.1;
        double expectedLoss = Math.Log(3) + (emptyTargets ? 0 : IndependentDetrBoxObjective(
            Enumerable.Repeat(boxCoordinate, 4).ToArray(), targetBox));
        double tolerance = typeof(T) == typeof(float) ? 2e-5 : 2e-8;
        Assert.InRange(Math.Abs(expectedLoss - ops.ToDouble(detector.GetLastLoss())), 0, tolerance);
        double parameterTolerance = typeof(T) == typeof(float) ? 2e-7 : 2e-8;
        for (int label = 0; label < 3; label++)
        {
            double targetMass = label == 2 ? (emptyTargets ? 50 : 49) * 0.1 : (label == 0 && !emptyTargets ? 1 : 0);
            double expected = -learningRate * (1.0 / 3 - targetMass / denominator);
            Assert.InRange(Math.Abs(expected - ops.ToDouble(classBias[label])), 0, parameterTolerance);
        }
        for (int coordinate = 0; coordinate < 4; coordinate++)
        {
            var plus = Enumerable.Repeat(boxCoordinate, 4).ToArray();
            var minus = Enumerable.Repeat(boxCoordinate, 4).ToArray();
            const double epsilon = 1e-6;
            plus[coordinate] += epsilon;
            minus[coordinate] -= epsilon;
            double derivative = emptyTargets ? 0 :
                (IndependentDetrBoxObjective(plus, targetBox) - IndependentDetrBoxObjective(minus, targetBox)) / (2 * epsilon);
            if (!emptyTargets) Assert.True(Math.Abs(derivative) > 0.1);
            // Semantic training applies sigmoid to the real RAW head. The derivative at bias
            // zero is 1/4; the nonzero-logit control also rejects accidentally applying it twice.
            double expected = actualLogit - learningRate * boxCoordinate * (1 - boxCoordinate) * derivative;
            Assert.InRange(Math.Abs(expected - ops.ToDouble(boxBias[coordinate])), 0, parameterTolerance);
        }
    }

    private static double IndependentDetrBoxObjective(double[] predicted, double[] target)
    {
        var p = new[] { predicted[0] - predicted[2] / 2, predicted[1] - predicted[3] / 2,
            predicted[0] + predicted[2] / 2, predicted[1] + predicted[3] / 2 };
        var t = new[] { target[0] - target[2] / 2, target[1] - target[3] / 2,
            target[0] + target[2] / 2, target[1] + target[3] / 2 };
        double intersection = Math.Max(0, Math.Min(p[2], t[2]) - Math.Max(p[0], t[0]))
            * Math.Max(0, Math.Min(p[3], t[3]) - Math.Max(p[1], t[1]));
        double union = predicted[2] * predicted[3] + target[2] * target[3] - intersection;
        double enclosure = (Math.Max(p[2], t[2]) - Math.Min(p[0], t[0]))
            * (Math.Max(p[3], t[3]) - Math.Min(p[1], t[1]));
        const double stabilityEpsilon = 1e-7; // Existing engine GIoU contract, not a loss correction.
        double giou = 1 - intersection / (union + stabilityEpsilon) + (enclosure - union) / (enclosure + stabilityEpsilon);
        double l1 = predicted.Zip(target, (left, right) => Math.Abs(left - right)).Sum();
        return 5 * l1 + 2 * giou;
    }

    /// <summary>Checks real forward, decode, confidence ordering, and suppression with known live heads.</summary>
    [Fact(Timeout = 120000)]
    public async Task Detect_ControlledPositiveHead_ShouldDecodeRankAndSuppressKnownCandidates()
    {
        await Task.Yield();
        using var arena = TensorArena.Create();
        using var detector = CreatePositiveObjectDetector(ObjectDetectionPositiveFixture<T>.CreateOptions());
        ObjectDetectionPositiveFixture<T>.Verify(detector);
    }

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

        // Compare against the threshold the detector says it applies: set-prediction detectors
        // (DETR, RT-DETR) declare a higher one rather than suppressing at the caller's value.
        double nmsThreshold = detector.EffectiveNmsThreshold(DetectNmsThreshold);
        var detections = detector.Detect(CreateRandomImage(rng), DetectConfidenceThreshold, DetectNmsThreshold).Detections;

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
                var (singleX1, singleY1, singleX2, singleY2) = single[d].Box.ToXYXY();
                var (batchX1, batchY1, batchX2, batchY2) = fromBatch[d].Box.ToXYXY();
                Assert.Equal(singleX1, batchX1, 8);
                Assert.Equal(singleY1, batchY1, 8);
                Assert.Equal(singleX2, batchX2, 8);
                Assert.Equal(singleY2, batchY2, 8);
            }
        }
    }

    private Tensor<T> ExtractImage(Tensor<T> batch, int index)
    {
        var shape = (int[])batch._shape.Clone();
        shape[0] = 1;

        int stride = 1;
        for (int d = 1; d < shape.Length; d++)
        {
            stride *= shape[d];
        }

        var image = new Tensor<T>(shape);
        for (int i = 0; i < stride; i++)
        {
            image[i] = batch[(index * stride) + i];
        }

        return image;
    }

    [Fact(Timeout = 180000)]
    public async Task Detect_BoxesShouldLieInsideASourceImageOfAnotherSize()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var detector = CreateDetector();

        // A source image whose size and aspect ratio differ from the network input. Boxes decode in
        // the network-input frame and must be mapped back to this frame; clipping them without that
        // mapping produced inverted boxes (x1 beyond x2) and silently dropped the rest.
        int height = InputShape[2] * 3 / 4, width = InputShape[3] * 5 / 4;
        var image = new Tensor<T>(new[] { 1, InputShape[1], height, width });
        for (int i = 0; i < image.Length; i++)
        {
            image[i] = ToT(rng.NextDouble());
        }

        var result = detector.Detect(image, 0.0, DetectNmsThreshold);

        Assert.Equal(width, result.ImageWidth);
        Assert.Equal(height, result.ImageHeight);
        foreach (var detection in result.Detections)
        {
            var (xMin, yMin, xMax, yMax) = detection.Box.ToXYXY();
            Assert.True(xMax > xMin && yMax > yMin, $"Degenerate box ({xMin},{yMin})-({xMax},{yMax}).");
            Assert.InRange(xMin, -1e-6, width + 1e-6);
            Assert.InRange(xMax, -1e-6, width + 1e-6);
            Assert.InRange(yMin, -1e-6, height + 1e-6);
            Assert.InRange(yMax, -1e-6, height + 1e-6);
        }
    }
}

/// <summary>Default-precision alias used by the generated fixtures.</summary>
public abstract class ObjectDetectionTestBase : ObjectDetectionTestBase<double> { }
