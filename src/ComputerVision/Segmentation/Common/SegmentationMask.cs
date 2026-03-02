namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Represents a single segmentation mask with associated metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A segmentation mask is a binary image where each pixel is either
/// part of the object (1) or not (0). This class wraps a mask with useful metadata like
/// the object's class, confidence score, area, and bounding box.
/// </para>
/// </remarks>
public class SegmentationMask<T>
{
    /// <summary>
    /// Binary mask tensor [H, W] where values > 0 indicate the segmented region.
    /// </summary>
    public Tensor<T> Mask { get; set; }

    /// <summary>
    /// Class ID of the segmented object.
    /// </summary>
    public int ClassId { get; set; }

    /// <summary>
    /// Class name (if available).
    /// </summary>
    public string? ClassName { get; set; }

    /// <summary>
    /// Confidence score in [0, 1].
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Instance ID (unique per instance for instance/panoptic segmentation).
    /// </summary>
    public int InstanceId { get; set; }

    /// <summary>
    /// Object tracking ID (for video segmentation across frames).
    /// </summary>
    public int TrackingId { get; set; } = -1;

    /// <summary>
    /// Bounding box [x1, y1, x2, y2] in pixel coordinates.
    /// </summary>
    public double[]? BoundingBox { get; set; }

    /// <summary>
    /// Area of the mask in pixels.
    /// </summary>
    public int Area { get; set; }

    /// <summary>
    /// Stability score measuring mask consistency under perturbations (SAM-style).
    /// </summary>
    public double StabilityScore { get; set; }

    /// <summary>
    /// IoU prediction score (if the model supports it).
    /// </summary>
    public double PredictedIoU { get; set; }

    /// <summary>
    /// Centroid (x, y) of the mask.
    /// </summary>
    public (double X, double Y)? Centroid { get; set; }

    /// <summary>
    /// Whether this mask represents a "thing" (countable object) or "stuff" (amorphous region).
    /// </summary>
    public bool IsThing { get; set; } = true;

    /// <summary>
    /// Creates a new segmentation mask.
    /// </summary>
    /// <param name="mask">Binary mask tensor [H, W].</param>
    /// <param name="classId">Class ID of the segmented object.</param>
    /// <param name="score">Confidence score in [0, 1].</param>
    public SegmentationMask(Tensor<T> mask, int classId, double score)
    {
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (classId < 0) throw new ArgumentOutOfRangeException(nameof(classId), "Class ID must be non-negative.");
        if (score < 0 || score > 1) throw new ArgumentOutOfRangeException(nameof(score), "Score must be in [0, 1].");

        Mask = mask;
        ClassId = classId;
        Score = score;
    }
}
