using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Helpers;

namespace AiDotNet.Metrics;

/// <summary>
/// COCO-style evaluation metrics for object detection: Average Precision (AP),
/// mean Average Precision (mAP) and the underlying precision-recall curve.
/// </summary>
/// <remarks>
/// <para>
/// These are the metrics the object-detection literature reports. A detector emits a set of
/// detections per image, each carrying a box, a class and a confidence score. Evaluation matches
/// those predictions against ground-truth detections and summarises the quality of the ranking.
/// Ground truth is expressed with the same <see cref="Detection{T}"/> type, of which only
/// <see cref="Detection{T}.Box"/> and <see cref="Detection{T}.ClassId"/> are read - its confidence
/// is ignored.
/// </para>
/// <para><b>The matching rule (COCO / Pascal VOC):</b>
/// predictions for one class are sorted by confidence, highest first. Each prediction is
/// matched greedily to the highest-IoU ground-truth box of the same class in the same image
/// that has not already been claimed. A match with IoU at or above the threshold is a true
/// positive; anything else is a false positive. Ground-truth boxes that end up unmatched are
/// false negatives. This one-to-one, highest-confidence-wins rule is what stops a detector
/// from inflating its score by emitting many overlapping boxes for the same object.
/// </para>
/// <para><b>Interpolation:</b>
/// AP is the area under the precision-recall curve. COCO computes it by sampling the curve at
/// 101 evenly spaced recall levels (0.00, 0.01, ... 1.00) and, at each level, taking the highest
/// precision observed at that recall or beyond. That highest-precision-at-or-beyond step removes
/// the small downward wiggles the raw curve has, which would otherwise make the score depend on
/// ties in the confidence ordering.
/// </para>
/// <para><b>Which number to report:</b>
/// <list type="bullet">
/// <item><description><see cref="MeanAveragePrecision"/> at 0.5 is Pascal VOC mAP@0.5 - the lenient,
/// widely quoted number.</description></item>
/// <item><description><see cref="MeanAveragePrecisionRange"/> is COCO mAP@[.50:.95], the primary
/// COCO metric: mAP averaged over ten IoU thresholds from 0.50 to 0.95. It rewards precise
/// localisation, so it is always lower than mAP@0.5.</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> IoU (intersection over union) measures how much a predicted box
/// overlaps a real one: 1.0 means identical, 0.0 means no overlap at all. An IoU threshold of 0.5
/// says a prediction counts as correct if it covers at least half the union of itself and the
/// real box. Average Precision then rolls the whole precision/recall trade-off into a single
/// number between 0 and 1, where higher is better. Reporting mAP@[.50:.95] rather than mAP@0.5
/// is the stricter, modern convention because it also asks the box to be tightly placed, not
/// merely in roughly the right spot.
/// </para>
/// <example>
/// <code>
/// var metrics = new ObjectDetectionMetrics&lt;double&gt;();
/// var predicted = images.Select(img =&gt; detector.Detect(img).Detections).ToList();
/// double cocoMap = metrics.MeanAveragePrecisionRange(predicted, groundTruthPerImage);
/// double vocMap = metrics.MeanAveragePrecision(predicted, groundTruthPerImage, 0.5);
/// </code>
/// </example>
/// </remarks>
/// <typeparam name="T">The numeric type the detections are expressed in.</typeparam>
public class ObjectDetectionMetrics<T> where T : struct
{
    /// <summary>
    /// Number of recall points COCO samples the precision-recall curve at (0.00 to 1.00 inclusive).
    /// </summary>
    private const int RecallSampleCount = 101;

    // Bound per-threshold claims and AP points even for very densely sampled ranges. COCO's
    // ten thresholds fit in one batch; larger ranges reuse class preparation across batches.
    private const int ThresholdBatchSize = 32;

    /// <summary>
    /// The numeric operations provider for type <typeparamref name="T"/>.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the <see cref="ObjectDetectionMetrics{T}"/> class.
    /// </summary>
    public ObjectDetectionMetrics()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Computes Average Precision for a single class at one IoU threshold.
    /// </summary>
    /// <param name="predictions">Predicted detections, one list per image. Confidence drives the ranking.</param>
    /// <param name="groundTruth">Ground-truth detections, one list per image, aligned with
    /// <paramref name="predictions"/>. Only box and class are read.</param>
    /// <param name="classIndex">The class to score. Detections of other classes are ignored.</param>
    /// <param name="iouThreshold">Minimum IoU for a prediction to count as a true positive.</param>
    /// <returns>AP in [0, 1], or <see cref="double.NaN"/> when the class has no ground-truth boxes
    /// (an undefined score, which <see cref="MeanAveragePrecision"/> excludes from its average).</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The two lists describe a different number of images.</exception>
    public double AveragePrecision(
        IReadOnlyList<IReadOnlyList<Detection<T>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth,
        int classIndex,
        double iouThreshold = 0.5)
    {
        var curve = ComputeCurve(predictions, groundTruth, classIndex, iouThreshold, out int groundTruthCount);
        if (groundTruthCount == 0)
        {
            return double.NaN;
        }

        return InterpolatedAveragePrecision(curve.Precision, curve.Recall, curve.Precision.Length);
    }

    /// <summary>
    /// Computes mean Average Precision at one IoU threshold: the mean of the per-class
    /// <see cref="AveragePrecision"/> over every class that has at least one ground-truth box.
    /// </summary>
    /// <param name="predictions">Predicted detections, one list per image.</param>
    /// <param name="groundTruth">Ground-truth detections, one list per image.</param>
    /// <param name="iouThreshold">Minimum IoU for a prediction to count as a true positive. 0.5 is Pascal VOC mAP@0.5.</param>
    /// <returns>mAP in [0, 1], or 0 when the ground truth contains no detections at all.</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The two lists describe a different number of images.</exception>
    public double MeanAveragePrecision(
        IReadOnlyList<IReadOnlyList<Detection<T>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth,
        double iouThreshold = 0.5)
    {
        ValidateAligned(predictions, groundTruth);

        var classes = GetGroundTruthClasses(groundTruth);

        if (classes.Count == 0)
        {
            return 0.0;
        }

        double sum = 0.0;
        int counted = 0;
        foreach (int classIndex in classes)
        {
            double ap = AveragePrecision(predictions, groundTruth, classIndex, iouThreshold);
            if (!double.IsNaN(ap))
            {
                sum += ap;
                counted++;
            }
        }

        return counted > 0 ? sum / counted : 0.0;
    }

    /// <summary>
    /// Computes COCO mAP@[.50:.95]: <see cref="MeanAveragePrecision"/> averaged over a range of
    /// IoU thresholds. This is the primary COCO detection metric.
    /// </summary>
    /// <remarks>
    /// Ground-truth lists and stable confidence rankings are prepared once per class. Each batch
    /// of at most 32 thresholds has independent greedy matches and shares a lazily computed IoU
    /// row for the current prediction. Thus COCO's ten thresholds compute each needed IoU once;
    /// ranges spanning several batches may compute it once per batch. No all-pairs IoU matrix or
    /// range-sized collection of matching states is allocated. AP retains only true-positive
    /// points, bounding per-batch state by the number of ground-truth boxes, not false positives.
    /// </remarks>
    /// <param name="predictions">Predicted detections, one list per image.</param>
    /// <param name="groundTruth">Ground-truth detections, one list per image.</param>
    /// <param name="minIoU">First IoU threshold. COCO uses 0.50.</param>
    /// <param name="maxIoU">Last IoU threshold, inclusive. COCO uses 0.95.</param>
    /// <param name="step">Spacing between thresholds. COCO uses 0.05, giving ten thresholds.</param>
    /// <returns>mAP averaged across the thresholds, in [0, 1].</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="step"/> is not finite and positive,
    /// the range is non-finite, empty or outside [0, 1], or its threshold count exceeds <see cref="int.MaxValue"/>.</exception>
    public double MeanAveragePrecisionRange(
        IReadOnlyList<IReadOnlyList<Detection<T>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth,
        double minIoU = 0.5,
        double maxIoU = 0.95,
        double step = 0.05)
    {
        if (double.IsNaN(step) || double.IsInfinity(step) || step <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(step), step, "IoU step must be finite and positive.");
        }

        if (!IsUnitInterval(minIoU) || !IsUnitInterval(maxIoU) || minIoU > maxIoU)
        {
            throw new ArgumentOutOfRangeException(
                nameof(minIoU), $"IoU range [{minIoU}, {maxIoU}] must be non-empty and within [0, 1].");
        }

        // Derive the count first rather than accumulating threshold += step, so floating-point
        // drift cannot silently drop or duplicate the final threshold.
        double lastThresholdIndex = Math.Floor(((maxIoU - minIoU) / step) + 1e-9);
        if (lastThresholdIndex >= int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(step), step,
                "IoU step produces more thresholds than an Int32 count can represent.");
        }

        int thresholdCount = (int)lastThresholdIndex + 1;
        ValidateAligned(predictions, groundTruth);
        var preparedClasses = GetGroundTruthClasses(groundTruth)
            .Select(classIndex => PrepareClass(predictions, groundTruth, classIndex)).ToArray();
        int counted = preparedClasses.Count(prepared => prepared.GroundTruthCount > 0);
        if (counted == 0)
        {
            return 0.0;
        }

        double sum = 0.0;
        int firstThreshold = 0;
        while (firstThreshold < thresholdCount)
        {
            int batchCount = Math.Min(ThresholdBatchSize, thresholdCount - firstThreshold);
            var thresholds = new double[batchCount];
            var classSums = new double[batchCount];
            for (int i = 0; i < batchCount; i++)
            {
                thresholds[i] = minIoU + ((firstThreshold + i) * step);
            }

            foreach (var prepared in preparedClasses)
            {
                if (prepared.GroundTruthCount == 0)
                {
                    continue;
                }

                var scores = ComputeAveragePrecisionBatch(prepared, thresholds);
                for (int i = 0; i < batchCount; i++)
                {
                    classSums[i] += scores[i];
                }
            }

            // Preserve the original order of both sums: sorted classes within each threshold,
            // then increasing thresholds. Reordering these averages changes floating-point bits.
            for (int i = 0; i < batchCount; i++)
            {
                sum += classSums[i] / counted;
            }

            firstThreshold += batchCount;
        }

        return sum / thresholdCount;
    }

    // Both comparisons are false for NaN; infinities also fall outside this finite interval.
    private static bool IsUnitInterval(double value) => value >= 0.0 && value <= 1.0;

    /// <summary>
    /// Computes the raw (uninterpolated) precision-recall curve for one class, in descending
    /// confidence order. Point <c>i</c> is the precision and recall achieved when the top
    /// <c>i + 1</c> predictions are accepted.
    /// </summary>
    /// <param name="predictions">Predicted detections, one list per image.</param>
    /// <param name="groundTruth">Ground-truth detections, one list per image.</param>
    /// <param name="classIndex">The class to score.</param>
    /// <param name="iouThreshold">Minimum IoU for a prediction to count as a true positive.</param>
    /// <returns>Parallel precision and recall arrays. Both are empty when the class has no predictions.</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">The two lists describe a different number of images.</exception>
    public (double[] Precision, double[] Recall) PrecisionRecallCurve(
        IReadOnlyList<IReadOnlyList<Detection<T>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth,
        int classIndex,
        double iouThreshold = 0.5)
        => ComputeCurve(predictions, groundTruth, classIndex, iouThreshold, out _);

    private (double[] Precision, double[] Recall) ComputeCurve(
        IReadOnlyList<IReadOnlyList<Detection<T>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth,
        int classIndex,
        double iouThreshold,
        out int groundTruthCount)
    {
        ValidateAligned(predictions, groundTruth);

        var prepared = PrepareClass(predictions, groundTruth, classIndex);
        groundTruthCount = prepared.GroundTruthCount;
        var claimed = new bool[groundTruthCount];
        var precision = new double[prepared.RankOrder.Length];
        var recall = new double[precision.Length];
        int truePositives = 0;

        for (int rank = 0; rank < prepared.RankOrder.Length; rank++)
        {
            var (imageIndex, box) = prepared.Predictions[prepared.RankOrder[rank]];
            var candidates = prepared.TruthByImage[imageIndex];
            int offset = prepared.TruthOffsets[imageIndex];

            double bestIoU = 0.0;
            int bestCandidate = -1;
            for (int c = 0; c < candidates.Count; c++)
            {
                if (claimed[offset + c])
                {
                    continue;
                }

                double iou = box.IoU(candidates[c]);
                if (iou > bestIoU)
                {
                    bestIoU = iou;
                    bestCandidate = c;
                }
            }

            if (bestCandidate >= 0 && bestIoU >= iouThreshold)
            {
                claimed[offset + bestCandidate] = true;
                truePositives++;
            }

            precision[rank] = truePositives / (double)(rank + 1);
            recall[rank] = groundTruthCount > 0 ? truePositives / (double)groundTruthCount : 0.0;
        }

        return (precision, recall);
    }

    private PreparedClass PrepareClass(
        IReadOnlyList<IReadOnlyList<Detection<T>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth,
        int classIndex)
    {
        // Keep original per-image candidate order, including equal-IoU tie precedence. Geometry
        // is deliberately not read here: an already-claimed candidate must remain unused.
        var truthByImage = new List<BoundingBox<T>>[groundTruth.Count];
        var truthOffsets = new int[groundTruth.Count];
        int groundTruthCount = 0;
        int maxTruthPerImage = 0;
        for (int i = 0; i < groundTruth.Count; i++)
        {
            var kept = new List<BoundingBox<T>>();
            var image = groundTruth[i];
            if (image is not null)
            {
                foreach (var detection in image)
                {
                    if (detection is not null && detection.ClassId == classIndex && detection.Box is not null)
                    {
                        kept.Add(detection.Box);
                    }
                }
            }

            truthByImage[i] = kept;
            truthOffsets[i] = groundTruthCount;
            groundTruthCount += kept.Count;
            maxTruthPerImage = Math.Max(maxTruthPerImage, kept.Count);
        }

        // Every prediction of this class across all images, ranked by confidence. OrderByDescending
        // is a stable sort, so equal-confidence predictions keep their original order and the curve
        // is reproducible run to run.
        var ranked = new List<(int ImageIndex, BoundingBox<T> Box)>();
        var scores = new List<double>();
        for (int i = 0; i < predictions.Count; i++)
        {
            var image = predictions[i];
            if (image is null)
            {
                continue;
            }

            foreach (var detection in image)
            {
                if (detection is not null && detection.ClassId == classIndex && detection.Box is not null)
                {
                    ranked.Add((i, detection.Box));
                    scores.Add(_numOps.ToDouble(detection.Confidence));
                }
            }
        }

        var order = Enumerable.Range(0, ranked.Count).OrderByDescending(i => scores[i]).ToArray();

        return new PreparedClass(truthByImage, truthOffsets, ranked, order, groundTruthCount, maxTruthPerImage);
    }

    private static double[] ComputeAveragePrecisionBatch(PreparedClass prepared, double[] thresholds)
    {
        int maxPoints = Math.Min(prepared.RankOrder.Length, prepared.GroundTruthCount);
        var scores = new double[thresholds.Length];
        if (maxPoints == 0)
        {
            return scores;
        }

        var claimed = new bool[thresholds.Length][];
        var precision = new double[thresholds.Length][];
        var recall = new double[thresholds.Length][];
        var truePositives = new int[thresholds.Length];
        for (int threshold = 0; threshold < thresholds.Length; threshold++)
        {
            claimed[threshold] = new bool[prepared.GroundTruthCount];
            precision[threshold] = new double[maxPoints];
            recall[threshold] = new double[maxPoints];
        }

        // One lazily populated IoU row, not a predictions-by-ground-truth matrix. Rank stamps
        // distinguish uncomputed entries from every possible IoU value, including zero and NaN.
        var iouRow = new double[prepared.MaxTruthPerImage];
        var rowRanks = new int[prepared.MaxTruthPerImage];
        for (int rank = 0; rank < prepared.RankOrder.Length; rank++)
        {
            var (imageIndex, box) = prepared.Predictions[prepared.RankOrder[rank]];
            var candidates = prepared.TruthByImage[imageIndex];
            int offset = prepared.TruthOffsets[imageIndex];
            for (int threshold = 0; threshold < thresholds.Length; threshold++)
            {
                var candidateClaimed = claimed[threshold];
                double bestIoU = 0.0;
                int bestCandidate = -1;
                for (int c = 0; c < candidates.Count; c++)
                {
                    if (candidateClaimed[offset + c])
                    {
                        continue;
                    }

                    if (rowRanks[c] != rank + 1)
                    {
                        iouRow[c] = box.IoU(candidates[c]);
                        rowRanks[c] = rank + 1;
                    }

                    double iou = iouRow[c];
                    if (iou > bestIoU)
                    {
                        bestIoU = iou;
                        bestCandidate = c;
                    }
                }

                if (bestCandidate >= 0 && bestIoU >= thresholds[threshold])
                {
                    candidateClaimed[offset + bestCandidate] = true;
                    int point = truePositives[threshold]++;
                    precision[threshold][point] = truePositives[threshold] / (double)(rank + 1);
                    recall[threshold][point] = truePositives[threshold] / (double)prepared.GroundTruthCount;
                }
            }
        }

        // A false positive cannot improve precision at unchanged recall; its preceding true
        // positive dominates it. Initial false positives are zero, and no true positives means
        // AP zero. Keeping only TP points therefore preserves the exact 101-sample envelope.
        for (int threshold = 0; threshold < thresholds.Length; threshold++)
        {
            scores[threshold] = InterpolatedAveragePrecision(
                precision[threshold], recall[threshold], truePositives[threshold]);
        }

        return scores;
    }

    private static SortedSet<int> GetGroundTruthClasses(IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth)
    {
        // Classes absent from ground truth have undefined recall and are not averaged in.
        var classes = new SortedSet<int>();
        foreach (var image in groundTruth)
        {
            if (image is null)
            {
                continue;
            }

            foreach (var detection in image)
            {
                if (detection is not null)
                {
                    classes.Add(detection.ClassId);
                }
            }
        }

        return classes;
    }

    private sealed class PreparedClass
    {
        public List<BoundingBox<T>>[] TruthByImage { get; }
        public int[] TruthOffsets { get; }
        public List<(int ImageIndex, BoundingBox<T> Box)> Predictions { get; }
        public int[] RankOrder { get; }
        public int GroundTruthCount { get; }
        public int MaxTruthPerImage { get; }

        public PreparedClass(List<BoundingBox<T>>[] truthByImage, int[] truthOffsets,
            List<(int ImageIndex, BoundingBox<T> Box)> predictions, int[] rankOrder,
            int groundTruthCount, int maxTruthPerImage)
        {
            TruthByImage = truthByImage;
            TruthOffsets = truthOffsets;
            Predictions = predictions;
            RankOrder = rankOrder;
            GroundTruthCount = groundTruthCount;
            MaxTruthPerImage = maxTruthPerImage;
        }
    }

    /// <summary>
    /// Area under the precision-recall curve using COCO 101-point interpolation: at each of 101
    /// evenly spaced recall levels, take the highest precision attained at that recall or beyond,
    /// then average those 101 values.
    /// </summary>
    private static double InterpolatedAveragePrecision(double[] precision, double[] recall, int pointCount)
    {
        if (pointCount == 0)
        {
            return 0.0;
        }

        // Sweep right-to-left so envelope[i] is the best precision achievable at recall >= recall[i].
        var envelope = new double[pointCount];
        double running = 0.0;
        for (int i = pointCount - 1; i >= 0; i--)
        {
            running = Math.Max(running, precision[i]);
            envelope[i] = running;
        }

        double sum = 0.0;
        int cursor = 0;
        for (int s = 0; s < RecallSampleCount; s++)
        {
            double target = s / (double)(RecallSampleCount - 1);

            // recall is non-decreasing along the ranking, so the cursor only ever moves forward.
            while (cursor < pointCount && recall[cursor] < target)
            {
                cursor++;
            }

            if (cursor >= pointCount)
            {
                break; // No prediction reaches this recall; the remaining samples contribute 0.
            }

            sum += envelope[cursor];
        }

        return sum / RecallSampleCount;
    }

    private static void ValidateAligned(
        IReadOnlyList<IReadOnlyList<Detection<T>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth)
    {
        if (predictions is null)
        {
            throw new ArgumentNullException(nameof(predictions));
        }

        if (groundTruth is null)
        {
            throw new ArgumentNullException(nameof(groundTruth));
        }

        if (predictions.Count != groundTruth.Count)
        {
            throw new ArgumentException(
                $"Predictions cover {predictions.Count} images but ground truth covers {groundTruth.Count}. "
                + "Both lists must be indexed by the same image order.",
                nameof(predictions));
        }
    }
}
