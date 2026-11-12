using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;

namespace AiDotNet.ComputerVision.Detection.PostProcessing;

/// <summary>
/// Implements Distance-IoU based Non-Maximum Suppression for improved localization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DIoU-NMS is an improved version of standard NMS that considers
/// the distance between box centers, not just their overlap. This helps preserve nearby
/// objects that might be suppressed by standard NMS.</para>
/// <para>Reference: Zheng et al., "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression", AAAI 2020</para>
/// </remarks>
public class DIoUNMS<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly NMS<T> _nms;

    /// <summary>
    /// Creates a new DIoU-NMS instance.
    /// </summary>
    public DIoUNMS()
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _nms = new NMS<T>();
    }

    /// <summary>
    /// Applies DIoU-NMS to a list of detections.
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="diouThreshold">DIoU threshold for suppression (typically higher than IoU threshold).</param>
    /// <returns>Filtered list of detections.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> DIoU-NMS uses the distance between box centers as an additional
    /// criterion. Two boxes might have high IoU but if their centers are far apart, they might
    /// represent different objects and should both be kept.</para>
    /// </remarks>
    public List<Detection<T>> Apply(List<Detection<T>> detections, double diouThreshold)
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

            // Suppress all boxes with DIoU > threshold
            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (suppressed[j]) continue;

                double diou = _nms.ComputeDIoU(sorted[i].Box, sorted[j].Box);
                if (diou > diouThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }

    /// <summary>
    /// Applies class-aware DIoU-NMS.
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="diouThreshold">DIoU threshold for suppression.</param>
    /// <returns>Filtered list of detections with per-class DIoU-NMS applied.</returns>
    public List<Detection<T>> ApplyClassAware(List<Detection<T>> detections, double diouThreshold)
    {
        if (detections.Count == 0) return new List<Detection<T>>();

        var result = new List<Detection<T>>();

        // Group by class
        var byClass = detections.GroupBy(d => d.ClassId);

        foreach (var classGroup in byClass)
        {
            var classDetections = classGroup.ToList();
            var nmsResult = Apply(classDetections, diouThreshold);
            result.AddRange(nmsResult);
        }

        // Sort final result by confidence
        return result.OrderByDescending(d => _numOps.ToDouble(d.Confidence)).ToList();
    }

    /// <summary>
    /// Applies DIoU-NMS with adaptive threshold based on box density.
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="baseDiouThreshold">Base DIoU threshold.</param>
    /// <param name="densityFactor">How much to adjust threshold based on density (0.0 to 1.0).</param>
    /// <returns>Filtered list of detections.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> In crowded scenes with many overlapping objects, using a fixed
    /// threshold might suppress too many valid detections. Adaptive DIoU-NMS adjusts the threshold
    /// based on the local density of detections.</para>
    /// </remarks>
    public List<Detection<T>> ApplyAdaptive(
        List<Detection<T>> detections,
        double baseDiouThreshold,
        double densityFactor = 0.5)
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

            // Count nearby boxes to determine local density
            int nearbyCount = 0;
            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (suppressed[j]) continue;

                double iou = _nms.ComputeIoU(sorted[i].Box, sorted[j].Box);
                if (iou > 0.1) // Low threshold to count "nearby" boxes
                {
                    nearbyCount++;
                }
            }

            // Adjust threshold based on density
            double adaptiveThreshold = baseDiouThreshold;
            if (nearbyCount > 0)
            {
                // Higher density = higher threshold (more permissive)
                double densityAdjustment = Math.Min(nearbyCount * 0.05 * densityFactor, 0.2);
                adaptiveThreshold = Math.Min(baseDiouThreshold + densityAdjustment, 0.9);
            }

            // Suppress boxes above adaptive threshold
            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (suppressed[j]) continue;

                double diou = _nms.ComputeDIoU(sorted[i].Box, sorted[j].Box);
                if (diou > adaptiveThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return kept;
    }

    /// <summary>
    /// Applies batched DIoU-NMS for multiple images.
    /// </summary>
    /// <param name="batchDetections">Detections for each image in the batch.</param>
    /// <param name="diouThreshold">DIoU threshold for suppression.</param>
    /// <param name="classAware">Whether to apply class-aware NMS.</param>
    /// <returns>Filtered detections for each image.</returns>
    public List<List<Detection<T>>> ApplyBatched(
        List<List<Detection<T>>> batchDetections,
        double diouThreshold,
        bool classAware = true)
    {
        var results = new List<List<Detection<T>>>();

        foreach (var imageDetections in batchDetections)
        {
            var filtered = classAware
                ? ApplyClassAware(imageDetections, diouThreshold)
                : Apply(imageDetections, diouThreshold);
            results.Add(filtered);
        }

        return results;
    }
}
