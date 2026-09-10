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

        return InterpolatedAveragePrecision(curve.Precision, curve.Recall);
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

        // Only classes that actually occur in the ground truth are scored. A class the detector
        // hallucinates but that never appears has no defined recall, so averaging it in would be
        // meaningless; its false positives still suppress the precision of the classes it competes with.
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
    /// <param name="predictions">Predicted detections, one list per image.</param>
    /// <param name="groundTruth">Ground-truth detections, one list per image.</param>
    /// <param name="minIoU">First IoU threshold. COCO uses 0.50.</param>
    /// <param name="maxIoU">Last IoU threshold, inclusive. COCO uses 0.95.</param>
    /// <param name="step">Spacing between thresholds. COCO uses 0.05, giving ten thresholds.</param>
    /// <returns>mAP averaged across the thresholds, in [0, 1].</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="step"/> is not positive, or the
    /// range is empty or outside [0, 1].</exception>
    public double MeanAveragePrecisionRange(
        IReadOnlyList<IReadOnlyList<Detection<T>>> predictions,
        IReadOnlyList<IReadOnlyList<Detection<T>>> groundTruth,
        double minIoU = 0.5,
        double maxIoU = 0.95,
        double step = 0.05)
    {
        if (step <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(step), step, "IoU step must be positive.");
        }

        if (minIoU < 0.0 || maxIoU > 1.0 || minIoU > maxIoU)
        {
            throw new ArgumentOutOfRangeException(
                nameof(minIoU), $"IoU range [{minIoU}, {maxIoU}] must be non-empty and within [0, 1].");
        }

        // Derive the count first rather than accumulating threshold += step, so floating-point
        // drift cannot silently drop or duplicate the final threshold.
        int thresholdCount = (int)Math.Floor(((maxIoU - minIoU) / step) + 1e-9) + 1;

        double sum = 0.0;
        for (int i = 0; i < thresholdCount; i++)
        {
            sum += MeanAveragePrecision(predictions, groundTruth, minIoU + (i * step));
        }

        return sum / thresholdCount;
    }

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

        // Ground truth for this class, kept per image alongside a claimed flag so each real box
        // can satisfy at most one prediction.
        var truthByImage = new List<BoundingBox<T>>[groundTruth.Count];
        var claimed = new bool[groundTruth.Count][];
        groundTruthCount = 0;
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
            claimed[i] = new bool[kept.Count];
            groundTruthCount += kept.Count;
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

        var precision = new double[order.Length];
        var recall = new double[order.Length];
        int truePositives = 0;

        for (int rank = 0; rank < order.Length; rank++)
        {
            var (imageIndex, box) = ranked[order[rank]];
            var candidates = truthByImage[imageIndex];
            var candidateClaimed = claimed[imageIndex];

            double bestIoU = 0.0;
            int bestCandidate = -1;
            for (int c = 0; c < candidates.Count; c++)
            {
                if (candidateClaimed[c])
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
                candidateClaimed[bestCandidate] = true;
                truePositives++;
            }

            precision[rank] = truePositives / (double)(rank + 1);
            recall[rank] = groundTruthCount > 0 ? truePositives / (double)groundTruthCount : 0.0;
        }

        return (precision, recall);
    }

    /// <summary>
    /// Area under the precision-recall curve using COCO 101-point interpolation: at each of 101
    /// evenly spaced recall levels, take the highest precision attained at that recall or beyond,
    /// then average those 101 values.
    /// </summary>
    private static double InterpolatedAveragePrecision(double[] precision, double[] recall)
    {
        if (precision.Length == 0)
        {
            return 0.0;
        }

        // Sweep right-to-left so envelope[i] is the best precision achievable at recall >= recall[i].
        var envelope = new double[precision.Length];
        double running = 0.0;
        for (int i = precision.Length - 1; i >= 0; i--)
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
            while (cursor < recall.Length && recall[cursor] < target)
            {
                cursor++;
            }

            if (cursor >= recall.Length)
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
