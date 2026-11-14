using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;

namespace AiDotNet.ComputerVision.Detection.PostProcessing;

/// <summary>
/// Implements Non-Maximum Suppression (NMS) algorithms for removing duplicate detections.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> When an object detector runs, it often produces multiple
/// overlapping bounding boxes for the same object. NMS removes these duplicates by:
/// 1. Keeping the detection with highest confidence
/// 2. Removing other detections that overlap too much with it
/// 3. Repeating until all duplicates are removed
/// </para>
/// </remarks>
public class NMS<T>
{
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Creates a new NMS instance.
    /// </summary>
    public NMS()
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Performs standard NMS on a list of detections.
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="iouThreshold">IoU threshold above which boxes are considered duplicates.</param>
    /// <returns>Filtered list of detections.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the classic NMS algorithm:
    /// 1. Sort detections by confidence (highest first)
    /// 2. Take the top detection, add to results
    /// 3. Remove all detections that overlap with it by more than iouThreshold
    /// 4. Repeat until no detections remain
    /// </para>
    /// </remarks>
    public List<Detection<T>> Apply(List<Detection<T>> detections, double iouThreshold)
    {
        if (detections.Count == 0) return new List<Detection<T>>();

        // Sort by confidence (descending)
        var sorted = detections
            .OrderByDescending(d => _numOps.ToDouble(d.Confidence))
            .ToList();

        var kept = new List<Detection<T>>();
        var suppressed = new bool[sorted.Count];

        for (int i = 0; i < sorted.Count; i++)
        {
            if (suppressed[i]) continue;

            kept.Add(sorted[i]);

            // Suppress all boxes with IoU > threshold
            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (suppressed[j]) continue;

                double iou = ComputeIoU(sorted[i].Box, sorted[j].Box);
                if (iou > iouThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }

    /// <summary>
    /// Performs class-aware NMS (applies NMS separately per class).
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="iouThreshold">IoU threshold for suppression.</param>
    /// <returns>Filtered list of detections.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Class-aware NMS only suppresses boxes of the same class.
    /// This allows a "person" detection to overlap with a "car" detection without either
    /// being suppressed.</para>
    /// </remarks>
    public List<Detection<T>> ApplyClassAware(List<Detection<T>> detections, double iouThreshold)
    {
        if (detections.Count == 0) return new List<Detection<T>>();

        var result = new List<Detection<T>>();

        // Group by class
        var byClass = detections.GroupBy(d => d.ClassId);

        foreach (var classGroup in byClass)
        {
            var classDetections = classGroup.ToList();
            var nmsResult = Apply(classDetections, iouThreshold);
            result.AddRange(nmsResult);
        }

        // Sort final result by confidence
        return result.OrderByDescending(d => _numOps.ToDouble(d.Confidence)).ToList();
    }

    /// <summary>
    /// Performs batched NMS for efficient processing of multiple images.
    /// </summary>
    /// <param name="batchDetections">Detections grouped by image index.</param>
    /// <param name="iouThreshold">IoU threshold for suppression.</param>
    /// <param name="classAware">Whether to apply class-aware NMS.</param>
    /// <returns>Filtered detections for each image.</returns>
    public List<List<Detection<T>>> ApplyBatched(
        List<List<Detection<T>>> batchDetections,
        double iouThreshold,
        bool classAware = true)
    {
        var results = new List<List<Detection<T>>>();

        foreach (var imageDetections in batchDetections)
        {
            var filtered = classAware
                ? ApplyClassAware(imageDetections, iouThreshold)
                : Apply(imageDetections, iouThreshold);
            results.Add(filtered);
        }

        return results;
    }

    /// <summary>
    /// Computes Intersection over Union (IoU) between two bounding boxes.
    /// </summary>
    /// <param name="box1">First bounding box.</param>
    /// <param name="box2">Second bounding box.</param>
    /// <returns>IoU value between 0 and 1.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> IoU measures how much two boxes overlap:
    /// - IoU = 0: No overlap
    /// - IoU = 1: Boxes are identical
    /// - IoU = 0.5: About 50% overlap
    /// </para>
    /// </remarks>
    public double ComputeIoU(BoundingBox<T> box1, BoundingBox<T> box2)
    {
        // Convert both boxes to XYXY format
        var (x1Min, y1Min, x1Max, y1Max) = box1.ToXYXY();
        var (x2Min, y2Min, x2Max, y2Max) = box2.ToXYXY();

        // Get intersection coordinates
        double interX1 = Math.Max(x1Min, x2Min);
        double interY1 = Math.Max(y1Min, y2Min);
        double interX2 = Math.Min(x1Max, x2Max);
        double interY2 = Math.Min(y1Max, y2Max);

        // Compute intersection area
        double intersectionWidth = Math.Max(0, interX2 - interX1);
        double intersectionHeight = Math.Max(0, interY2 - interY1);
        double intersectionArea = intersectionWidth * intersectionHeight;

        if (intersectionArea <= 0) return 0.0;

        // Compute union area
        double area1 = box1.Area();
        double area2 = box2.Area();
        double unionArea = area1 + area2 - intersectionArea;

        if (unionArea <= 0) return 0.0;

        return intersectionArea / unionArea;
    }

    /// <summary>
    /// Computes Generalized IoU (GIoU) between two bounding boxes.
    /// </summary>
    /// <param name="box1">First bounding box.</param>
    /// <param name="box2">Second bounding box.</param>
    /// <returns>GIoU value between -1 and 1.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> GIoU is an improved version of IoU that also considers
    /// the smallest enclosing box. Unlike IoU, GIoU can be negative when boxes don't overlap
    /// but provides gradients even for non-overlapping boxes.</para>
    /// </remarks>
    public double ComputeGIoU(BoundingBox<T> box1, BoundingBox<T> box2)
    {
        double iou = ComputeIoU(box1, box2);

        // Convert both boxes to XYXY format
        var (x1Min, y1Min, x1Max, y1Max) = box1.ToXYXY();
        var (x2Min, y2Min, x2Max, y2Max) = box2.ToXYXY();

        // Get enclosing box coordinates
        double encX1 = Math.Min(x1Min, x2Min);
        double encY1 = Math.Min(y1Min, y2Min);
        double encX2 = Math.Max(x1Max, x2Max);
        double encY2 = Math.Max(y1Max, y2Max);

        // Enclosing box area
        double enclosingArea = (encX2 - encX1) * (encY2 - encY1);

        if (enclosingArea <= 0) return iou;

        // Union area
        double area1 = box1.Area();
        double area2 = box2.Area();

        // Intersection area
        double interX1 = Math.Max(x1Min, x2Min);
        double interY1 = Math.Max(y1Min, y2Min);
        double interX2 = Math.Min(x1Max, x2Max);
        double interY2 = Math.Min(y1Max, y2Max);
        double intersectionArea = Math.Max(0, interX2 - interX1) * Math.Max(0, interY2 - interY1);

        double unionArea = area1 + area2 - intersectionArea;

        // GIoU = IoU - (enclosing - union) / enclosing
        double giou = iou - (enclosingArea - unionArea) / enclosingArea;

        return giou;
    }

    /// <summary>
    /// Computes Distance IoU (DIoU) between two bounding boxes.
    /// </summary>
    /// <param name="box1">First bounding box.</param>
    /// <param name="box2">Second bounding box.</param>
    /// <returns>DIoU value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> DIoU adds a penalty based on the distance between
    /// box centers. This helps with faster convergence during training and better
    /// localization.</para>
    /// </remarks>
    public double ComputeDIoU(BoundingBox<T> box1, BoundingBox<T> box2)
    {
        double iou = ComputeIoU(box1, box2);

        // Get center coordinates using ToCXCYWH
        var (cx1, cy1, _, _) = box1.ToCXCYWH();
        var (cx2, cy2, _, _) = box2.ToCXCYWH();

        double centerDistSq = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);

        // Get XYXY for enclosing box calculation
        var (x1Min, y1Min, x1Max, y1Max) = box1.ToXYXY();
        var (x2Min, y2Min, x2Max, y2Max) = box2.ToXYXY();

        // Diagonal of enclosing box
        double encX1 = Math.Min(x1Min, x2Min);
        double encY1 = Math.Min(y1Min, y2Min);
        double encX2 = Math.Max(x1Max, x2Max);
        double encY2 = Math.Max(y1Max, y2Max);

        double diagonalSq = (encX2 - encX1) * (encX2 - encX1) + (encY2 - encY1) * (encY2 - encY1);

        if (diagonalSq <= 0) return iou;

        // DIoU = IoU - d^2 / c^2
        double diou = iou - centerDistSq / diagonalSq;

        return diou;
    }

    /// <summary>
    /// Computes Complete IoU (CIoU) between two bounding boxes.
    /// </summary>
    /// <param name="box1">First bounding box.</param>
    /// <param name="box2">Second bounding box.</param>
    /// <returns>CIoU value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> CIoU extends DIoU by also considering aspect ratio.
    /// This provides the best localization accuracy and is used in modern YOLO versions.</para>
    /// </remarks>
    public double ComputeCIoU(BoundingBox<T> box1, BoundingBox<T> box2)
    {
        double diou = ComputeDIoU(box1, box2);

        // Get dimensions using ToCXCYWH
        var (_, _, w1, h1) = box1.ToCXCYWH();
        var (_, _, w2, h2) = box2.ToCXCYWH();

        if (h1 <= 0 || h2 <= 0) return diou;

        double arctan1 = Math.Atan(w1 / h1);
        double arctan2 = Math.Atan(w2 / h2);

        double v = (4.0 / (Math.PI * Math.PI)) * Math.Pow(arctan1 - arctan2, 2);

        double iou = ComputeIoU(box1, box2);

        // Alpha factor
        double alpha = v / (1 - iou + v + 1e-7);

        // CIoU = DIoU - alpha * v
        double ciou = diou - alpha * v;

        return ciou;
    }
}

/// <summary>
/// Row-wise overlap measures between two sets of XYXY boxes <c>[N, 4]</c>, as engine ops, with exactly the
/// conventions of <see cref="NMS{T}.ComputeIoU"/>, <see cref="NMS{T}.ComputeGIoU"/>,
/// <see cref="NMS{T}.ComputeDIoU"/> and <see cref="NMS{T}.ComputeCIoU"/> on XYXY boxes.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class BoxOverlap<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;

    private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    private readonly struct Corners
    {
        public Corners(Tensor<T> boxes)
        {
            int n = boxes.Shape[0];
            Tensor<T> Column(int index)
                => Engine.Reshape(Engine.TensorSlice(boxes, new[] { 0, index }, new[] { n, 1 }), new[] { n });
            X1 = Column(0); Y1 = Column(1); X2 = Column(2); Y2 = Column(3);
        }

        public Tensor<T> X1 { get; }
        public Tensor<T> Y1 { get; }
        public Tensor<T> X2 { get; }
        public Tensor<T> Y2 { get; }
    }

    /// <summary>Rows of a flat run of <c>4N</c> coordinates (or the first four columns of <c>[..., k]</c>) as <c>[N, 4]</c>.</summary>
    public static Tensor<T> AsRows(Tensor<T> boxes)
    {
        int last = boxes.Shape[boxes.Rank - 1];
        if (boxes.Rank == 1)
            return Engine.Reshape(boxes, new[] { boxes.Length / 4, 4 });
        int rows = boxes.Length / last;
        var flat = Engine.Reshape(boxes, new[] { rows, last });
        return last == 4 ? flat : Engine.TensorSlice(flat, new[] { 0, 0 }, new[] { rows, 4 });
    }

    /// <summary>
    /// IoU per row: 0 when the intersection or the union is not positive; areas clamp negative extents to zero.
    /// </summary>
    public static Tensor<T> IoU(Tensor<T> a, Tensor<T> b)
    {
        var (ca, cb) = ScaledCorners(a, b);
        return IoUParts(ca, cb).IoU;
    }

    /// <summary>GIoU per row: <c>IoU - (C - U) / C</c>, or IoU itself when the enclosing area C is not positive.</summary>
    public static Tensor<T> GIoU(Tensor<T> a, Tensor<T> b)
    {
        var (ca, cb) = ScaledCorners(a, b);
        var (iou, union) = IoUParts(ca, cb);
        var enclosing = Engine.TensorMultiply(
            Engine.TensorSubtract(Engine.TensorMax(ca.X2, cb.X2), Engine.TensorMin(ca.X1, cb.X1)),
            Engine.TensorSubtract(Engine.TensorMax(ca.Y2, cb.Y2), Engine.TensorMin(ca.Y1, cb.Y1)));
        var penalty = Ratio(Engine.TensorSubtract(enclosing, union), enclosing);
        return Engine.TensorSubtract(iou, penalty);
    }

    /// <summary>DIoU per row: <c>IoU - d^2 / c^2</c> (centre distance over enclosing diagonal), or IoU when c^2 is not positive.</summary>
    public static Tensor<T> DIoU(Tensor<T> a, Tensor<T> b)
    {
        var (ca, cb) = ScaledCorners(a, b);
        return DIoU(ca, cb, IoUParts(ca, cb).IoU);
    }

    /// <summary>
    /// CIoU per row: <c>DIoU - alpha v</c> with <c>v = 4 / pi^2 (atan(w1 / h1) - atan(w2 / h2))^2</c> and
    /// <c>alpha = v / (1 - IoU + v + 1e-7)</c>; DIoU itself when either height is not positive.
    /// </summary>
    public static Tensor<T> CIoU(Tensor<T> a, Tensor<T> b)
    {
        var (ca, cb) = ScaledCorners(a, b);
        var iou = IoUParts(ca, cb).IoU;
        var diou = DIoU(ca, cb, iou);

        var w1 = Engine.TensorSubtract(ca.X2, ca.X1);
        var h1 = Engine.TensorSubtract(ca.Y2, ca.Y1);
        var w2 = Engine.TensorSubtract(cb.X2, cb.X1);
        var h2 = Engine.TensorSubtract(cb.Y2, cb.Y1);
        var bothPositive = Engine.TensorMultiply(Engine.TensorGreaterThan(h1, NumOps.Zero), Engine.TensorGreaterThan(h2, NumOps.Zero));
        var ones = Engine.TensorOnesLike(h1);
        var safeH1 = Engine.TensorWhere(bothPositive, h1, ones);
        var safeH2 = Engine.TensorWhere(bothPositive, h2, ones);
        var angle = Engine.TensorSubtract(Engine.TensorAtan(Engine.TensorDivide(w1, safeH1)), Engine.TensorAtan(Engine.TensorDivide(w2, safeH2)));
        var v = Engine.TensorMultiplyScalar(Engine.TensorSquare(angle), NumOps.FromDouble(4.0 / (Math.PI * Math.PI)));
        var alpha = Engine.TensorDivide(v, Engine.TensorAddScalar(
            Engine.TensorAdd(Engine.ScalarMinusTensor(NumOps.One, iou), v), NumOps.FromDouble(1e-7)));
        var ciou = Engine.TensorSubtract(diou, Engine.TensorMultiply(alpha, v));
        return Engine.TensorWhere(bothPositive, ciou, diou);
    }

    /// <summary>
    /// The corners of both box sets, each row pair divided by its largest absolute coordinate.
    /// </summary>
    /// <remarks>
    /// Every measure here is a ratio of areas or of squared lengths, or an angle, so it is unchanged by scaling a row
    /// pair. Scaling to <c>[-1, 1]</c> keeps the areas and squared diagonals finite in half precision, where pixel
    /// boxes overflow 65504 (a 256 x 256 box already does) and the overflowing ratios became NaN.
    /// </remarks>
    private static (Corners A, Corners B) ScaledCorners(Tensor<T> a, Tensor<T> b)
    {
        int n = a.Shape[0];
        var largest = Engine.ReduceMax(Engine.TensorAbs(Engine.TensorConcatenate(new[] { a, b }, 1)), new[] { 1 }, keepDims: true);
        var safe = Engine.TensorWhere(Engine.TensorGreaterThan(largest, NumOps.Zero), largest, Engine.TensorOnesLike(largest));
        var divisor = Engine.TensorBroadcastTo(safe, new[] { n, 4 });
        return (new Corners(Engine.TensorDivide(a, divisor)), new Corners(Engine.TensorDivide(b, divisor)));
    }

    private static Tensor<T> DIoU(Corners ca, Corners cb, Tensor<T> iou)
    {
        var two = NumOps.FromDouble(2);
        var cx1 = Engine.TensorAdd(ca.X1, Engine.TensorDivideScalar(Engine.TensorSubtract(ca.X2, ca.X1), two));
        var cy1 = Engine.TensorAdd(ca.Y1, Engine.TensorDivideScalar(Engine.TensorSubtract(ca.Y2, ca.Y1), two));
        var cx2 = Engine.TensorAdd(cb.X1, Engine.TensorDivideScalar(Engine.TensorSubtract(cb.X2, cb.X1), two));
        var cy2 = Engine.TensorAdd(cb.Y1, Engine.TensorDivideScalar(Engine.TensorSubtract(cb.Y2, cb.Y1), two));
        var centerDistSq = Engine.TensorAdd(
            Engine.TensorSquare(Engine.TensorSubtract(cx1, cx2)), Engine.TensorSquare(Engine.TensorSubtract(cy1, cy2)));
        var diagonalSq = Engine.TensorAdd(
            Engine.TensorSquare(Engine.TensorSubtract(Engine.TensorMax(ca.X2, cb.X2), Engine.TensorMin(ca.X1, cb.X1))),
            Engine.TensorSquare(Engine.TensorSubtract(Engine.TensorMax(ca.Y2, cb.Y2), Engine.TensorMin(ca.Y1, cb.Y1))));
        return Engine.TensorSubtract(iou, Ratio(centerDistSq, diagonalSq));
    }

    private static (Tensor<T> IoU, Tensor<T> Union) IoUParts(Corners a, Corners b)
    {
        var interW = Engine.TensorMax(Engine.TensorSubtract(Engine.TensorMin(a.X2, b.X2), Engine.TensorMax(a.X1, b.X1)), NumOps.Zero);
        var interH = Engine.TensorMax(Engine.TensorSubtract(Engine.TensorMin(a.Y2, b.Y2), Engine.TensorMax(a.Y1, b.Y1)), NumOps.Zero);
        var intersection = Engine.TensorMultiply(interW, interH);
        var areaA = Engine.TensorMultiply(
            Engine.TensorMax(Engine.TensorSubtract(a.X2, a.X1), NumOps.Zero), Engine.TensorMax(Engine.TensorSubtract(a.Y2, a.Y1), NumOps.Zero));
        var areaB = Engine.TensorMultiply(
            Engine.TensorMax(Engine.TensorSubtract(b.X2, b.X1), NumOps.Zero), Engine.TensorMax(Engine.TensorSubtract(b.Y2, b.Y1), NumOps.Zero));
        var union = Engine.TensorSubtract(Engine.TensorAdd(areaA, areaB), intersection);
        var overlapping = Engine.TensorGreaterThan(intersection, NumOps.Zero);
        var iou = Engine.TensorMultiply(overlapping, Ratio(intersection, union));
        return (iou, union);
    }

    /// <summary><c>numerator / denominator</c> where the denominator is positive, else zero.</summary>
    private static Tensor<T> Ratio(Tensor<T> numerator, Tensor<T> denominator)
    {
        var positive = Engine.TensorGreaterThan(denominator, NumOps.Zero);
        var safe = Engine.TensorWhere(positive, denominator, Engine.TensorOnesLike(denominator));
        return Engine.TensorMultiply(positive, Engine.TensorDivide(numerator, safe));
    }
}
