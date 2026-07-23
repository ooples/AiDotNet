using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;

namespace AiDotNet.ComputerVision.Detection.PostProcessing;

/// <summary>
/// Implements Soft-NMS algorithm which reduces confidence of overlapping boxes instead of removing them.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Standard NMS completely removes overlapping boxes, which can be
/// problematic when objects are close together or occluded. Soft-NMS instead reduces the
/// confidence score of overlapping boxes, allowing them to potentially survive if their
/// confidence remains above the threshold.</para>
/// <para>Reference: Bodla et al., "Soft-NMS -- Improving Object Detection With One Line of Code", ICCV 2017</para>
/// </remarks>
public class SoftNMS<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly NMS<T> _nms;

    /// <summary>
    /// Soft-NMS decay method.
    /// </summary>
    public enum DecayMethod
    {
        /// <summary>Linear decay: score = score * (1 - IoU).</summary>
        Linear,
        /// <summary>Gaussian decay: score = score * exp(-IoU^2 / sigma).</summary>
        Gaussian
    }

    /// <summary>
    /// Creates a new Soft-NMS instance.
    /// </summary>
    public SoftNMS()
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _nms = new NMS<T>();
    }

    /// <summary>
    /// Applies Soft-NMS with linear decay.
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="iouThreshold">IoU threshold for applying decay.</param>
    /// <param name="scoreThreshold">Minimum score to keep a detection.</param>
    /// <returns>Filtered list of detections with adjusted confidences.</returns>
    public List<Detection<T>> ApplyLinear(
        List<Detection<T>> detections,
        double iouThreshold,
        double scoreThreshold = 0.001)
    {
        return Apply(detections, iouThreshold, scoreThreshold, DecayMethod.Linear, 0.5);
    }

    /// <summary>
    /// Applies Soft-NMS with Gaussian decay.
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="sigma">Gaussian decay parameter (higher = slower decay).</param>
    /// <param name="scoreThreshold">Minimum score to keep a detection.</param>
    /// <returns>Filtered list of detections with adjusted confidences.</returns>
    public List<Detection<T>> ApplyGaussian(
        List<Detection<T>> detections,
        double sigma = 0.5,
        double scoreThreshold = 0.001)
    {
        return Apply(detections, 1.0, scoreThreshold, DecayMethod.Gaussian, sigma);
    }

    /// <summary>
    /// Applies Soft-NMS with configurable decay method.
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="iouThreshold">IoU threshold (for linear) or ignored (for Gaussian).</param>
    /// <param name="scoreThreshold">Minimum score to keep a detection.</param>
    /// <param name="method">Decay method to use.</param>
    /// <param name="sigma">Sigma parameter for Gaussian decay.</param>
    /// <returns>Filtered list of detections with adjusted confidences.</returns>
    public List<Detection<T>> Apply(
        List<Detection<T>> detections,
        double iouThreshold,
        double scoreThreshold,
        DecayMethod method,
        double sigma)
    {
        if (detections.Count == 0) return new List<Detection<T>>();

        // Create mutable copies with score tracking
        var boxes = new List<(Detection<T> det, double score)>();
        foreach (var det in detections)
        {
            boxes.Add((det, _numOps.ToDouble(det.Confidence)));
        }

        var kept = new List<Detection<T>>();

        while (boxes.Count > 0)
        {
            // Find detection with max score
            int maxIdx = 0;
            double maxScore = boxes[0].score;
            for (int i = 1; i < boxes.Count; i++)
            {
                if (boxes[i].score > maxScore)
                {
                    maxScore = boxes[i].score;
                    maxIdx = i;
                }
            }

            // Extract the max detection
            var (maxDet, _) = boxes[maxIdx];
            boxes.RemoveAt(maxIdx);

            // Keep this detection if above threshold
            if (maxScore >= scoreThreshold)
            {
                // Create new detection with updated confidence
                var keptDet = new Detection<T>(
                    maxDet.Box,
                    maxDet.ClassId,
                    _numOps.FromDouble(maxScore),
                    maxDet.ClassName)
                {
                    Mask = maxDet.Mask,
                    Keypoints = maxDet.Keypoints,
                    TrackId = maxDet.TrackId
                };
                kept.Add(keptDet);
            }

            // Decay scores of remaining boxes based on IoU with max
            for (int i = 0; i < boxes.Count; i++)
            {
                var (det, score) = boxes[i];
                double iou = _nms.ComputeIoU(maxDet.Box, det.Box);

                double newScore = score;
                if (method == DecayMethod.Linear)
                {
                    if (iou > iouThreshold)
                    {
                        newScore = score * (1.0 - iou);
                    }
                }
                else // Gaussian
                {
                    newScore = score * Math.Exp(-(iou * iou) / sigma);
                }

                boxes[i] = (det, newScore);
            }

            // Remove boxes below score threshold
            boxes.RemoveAll(b => b.score < scoreThreshold);
        }

        return kept;
    }

    /// <summary>
    /// Applies class-aware Soft-NMS.
    /// </summary>
    /// <param name="detections">List of detections to filter.</param>
    /// <param name="iouThreshold">IoU threshold for linear decay.</param>
    /// <param name="scoreThreshold">Minimum score to keep.</param>
    /// <param name="method">Decay method.</param>
    /// <param name="sigma">Sigma for Gaussian decay.</param>
    /// <returns>Filtered detections with per-class Soft-NMS applied.</returns>
    public List<Detection<T>> ApplyClassAware(
        List<Detection<T>> detections,
        double iouThreshold = 0.5,
        double scoreThreshold = 0.001,
        DecayMethod method = DecayMethod.Gaussian,
        double sigma = 0.5)
    {
        if (detections.Count == 0) return new List<Detection<T>>();

        var result = new List<Detection<T>>();

        // Group by class
        var byClass = detections.GroupBy(d => d.ClassId);

        foreach (var classGroup in byClass)
        {
            var classDetections = classGroup.ToList();
            var nmsResult = Apply(classDetections, iouThreshold, scoreThreshold, method, sigma);
            result.AddRange(nmsResult);
        }

        // Sort final result by confidence
        return result.OrderByDescending(d => _numOps.ToDouble(d.Confidence)).ToList();
    }
}
