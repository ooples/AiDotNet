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
