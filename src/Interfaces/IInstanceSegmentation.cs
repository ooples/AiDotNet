using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for instance segmentation models that detect and mask individual object instances.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Instance segmentation combines object detection with pixel-level masking. Each detected
/// object gets its own binary mask, allowing you to distinguish between individual instances
/// of the same class (e.g., car #1 vs. car #2).
/// </para>
/// <para>
/// <b>For Beginners:</b> Instance segmentation answers "where is each individual object?"
///
/// Unlike semantic segmentation (which just says "these pixels are car"), instance segmentation
/// says "these pixels are car #1, those pixels are car #2, and those are car #3".
///
/// Each detection includes:
/// - A bounding box (rectangle around the object)
/// - A binary mask (exact pixel outline)
/// - A class label (what the object is)
/// - A confidence score (how sure the model is)
///
/// Models implementing this interface:
/// - YOLOv9-Seg, YOLO11-Seg, YOLOv12-Seg, YOLO26-Seg (real-time)
/// - Mask2Former, MaskDINO (transformer-based)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("InstanceSegmentation")]
public interface IInstanceSegmentation<T> : ISegmentationModel<T>
{
    /// <summary>
    /// Gets the maximum number of instances the model can detect per image.
    /// </summary>
    int MaxInstances { get; }

    /// <summary>
    /// Gets or sets the confidence threshold for filtering detections.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Only detections with confidence above this threshold are returned.
    /// Higher values (e.g., 0.7) give fewer but more confident detections.
    /// Lower values (e.g., 0.3) give more detections but some may be false positives.
    /// </para>
    /// </remarks>
    double ConfidenceThreshold { get; set; }

    /// <summary>
    /// Gets or sets the IoU threshold for non-maximum suppression.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When the model detects the same object multiple times,
    /// NMS removes duplicates. This threshold controls how much overlap is allowed
    /// before two detections are considered duplicates (default: 0.5).
    /// </para>
    /// </remarks>
    double NmsThreshold { get; set; }

    /// <summary>
    /// Detects instances and returns their masks, bounding boxes, and class labels.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>List of detected instances using the existing <see cref="InstanceSegmentationResult{T}"/> type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass in your image and get back a list of all detected objects,
    /// each with their exact pixel outline (mask), surrounding box, what they are (class),
    /// and how confident the model is about the detection.
    /// </para>
    /// </remarks>
    InstanceSegmentationResult<T> DetectInstances(Tensor<T> image);
}
