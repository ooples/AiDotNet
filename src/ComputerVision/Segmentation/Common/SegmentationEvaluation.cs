namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Comprehensive evaluation metrics for segmentation models covering all standard benchmarks.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These metrics measure how good a segmentation model is.
/// Different metrics matter for different tasks:
/// - mIoU: Standard for semantic segmentation (higher = better, max 100%)
/// - AP: Standard for instance segmentation (higher = better, max 100%)
/// - PQ: Standard for panoptic segmentation (higher = better, max 100%)
/// - Dice: Standard for medical segmentation (higher = better, max 1.0)
/// - J&amp;F: Standard for video object segmentation (higher = better, max 100%)
/// </para>
/// </remarks>
public class SegmentationEvaluation
{
    #region Semantic Segmentation Metrics

    /// <summary>
    /// Mean Intersection over Union across all classes. Primary metric for semantic segmentation.
    /// </summary>
    public double MeanIoU { get; set; }

    /// <summary>
    /// Per-class IoU scores.
    /// </summary>
    public Dictionary<int, double> PerClassIoU { get; set; } = [];

    /// <summary>
    /// Pixel accuracy: fraction of correctly classified pixels.
    /// </summary>
    public double PixelAccuracy { get; set; }

    /// <summary>
    /// Mean pixel accuracy across classes.
    /// </summary>
    public double MeanPixelAccuracy { get; set; }

    /// <summary>
    /// Frequency-weighted IoU (classes weighted by pixel frequency).
    /// </summary>
    public double FrequencyWeightedIoU { get; set; }

    #endregion

    #region Instance Segmentation Metrics

    /// <summary>
    /// Average Precision at IoU=0.50 (COCO-style AP50). Primary metric for instance segmentation.
    /// </summary>
    public double AP50 { get; set; }

    /// <summary>
    /// Average Precision at IoU=0.75 (stricter).
    /// </summary>
    public double AP75 { get; set; }

    /// <summary>
    /// Average Precision averaged over IoU thresholds 0.50:0.05:0.95 (COCO primary metric).
    /// </summary>
    public double AP { get; set; }

    /// <summary>
    /// AP for small objects (area &lt; 32^2 pixels).
    /// </summary>
    public double APSmall { get; set; }

    /// <summary>
    /// AP for medium objects (32^2 &lt; area &lt; 96^2 pixels).
    /// </summary>
    public double APMedium { get; set; }

    /// <summary>
    /// AP for large objects (area &gt; 96^2 pixels).
    /// </summary>
    public double APLarge { get; set; }

    /// <summary>
    /// Average Recall at maximum 1 detection per image.
    /// </summary>
    public double AR1 { get; set; }

    /// <summary>
    /// Average Recall at maximum 10 detections per image.
    /// </summary>
    public double AR10 { get; set; }

    /// <summary>
    /// Average Recall at maximum 100 detections per image.
    /// </summary>
    public double AR100 { get; set; }

    /// <summary>
    /// Per-class AP scores.
    /// </summary>
    public Dictionary<int, double> PerClassAP { get; set; } = [];

    #endregion

    #region Panoptic Segmentation Metrics

    /// <summary>
    /// Panoptic Quality. Primary metric for panoptic segmentation. PQ = SQ * RQ.
    /// </summary>
    public double PanopticQuality { get; set; }

    /// <summary>
    /// Segmentation Quality (average IoU of matched segments).
    /// </summary>
    public double SegmentationQuality { get; set; }

    /// <summary>
    /// Recognition Quality (F1 of matched segments, measures detection accuracy).
    /// </summary>
    public double RecognitionQuality { get; set; }

    /// <summary>
    /// PQ for "things" (countable objects) only.
    /// </summary>
    public double PQThings { get; set; }

    /// <summary>
    /// PQ for "stuff" (amorphous regions) only.
    /// </summary>
    public double PQStuff { get; set; }

    /// <summary>
    /// Per-class PQ scores.
    /// </summary>
    public Dictionary<int, double> PerClassPQ { get; set; } = [];

    #endregion

    #region Medical Segmentation Metrics

    /// <summary>
    /// Mean Dice score across classes. Primary metric for medical segmentation.
    /// </summary>
    public double MeanDice { get; set; }

    /// <summary>
    /// Per-class Dice scores.
    /// </summary>
    public Dictionary<int, double> PerClassDice { get; set; } = [];

    /// <summary>
    /// Mean Hausdorff Distance (95th percentile) in mm.
    /// </summary>
    public double MeanHD95 { get; set; }

    /// <summary>
    /// Per-class Hausdorff Distance (95th percentile).
    /// </summary>
    public Dictionary<int, double> PerClassHD95 { get; set; } = [];

    /// <summary>
    /// Normalized Surface Distance at specified tolerance.
    /// </summary>
    public double MeanNSD { get; set; }

    /// <summary>
    /// Average Symmetric Surface Distance in mm.
    /// </summary>
    public double MeanASSD { get; set; }

    /// <summary>
    /// Volume Similarity (1 = perfect, 0 = no overlap).
    /// </summary>
    public double MeanVolumeSimilarity { get; set; }

    #endregion

    #region Video Segmentation Metrics

    /// <summary>
    /// J score (Jaccard/IoU for VOS). Primary metric for video object segmentation.
    /// </summary>
    public double JScore { get; set; }

    /// <summary>
    /// F score (boundary F-measure for VOS).
    /// </summary>
    public double FScore { get; set; }

    /// <summary>
    /// J&amp;F mean (average of J and F scores). Primary overall VOS metric.
    /// </summary>
    public double JAndFMean { get; set; }

    /// <summary>
    /// Temporal stability score (how consistent masks are across frames).
    /// </summary>
    public double TemporalStability { get; set; }

    /// <summary>
    /// Per-object J&amp;F scores in video.
    /// </summary>
    public Dictionary<int, double> PerObjectJAndF { get; set; } = [];

    #endregion

    #region Boundary Metrics

    /// <summary>
    /// Boundary IoU at specified pixel tolerance.
    /// </summary>
    public double BoundaryIoU { get; set; }

    /// <summary>
    /// Boundary F1 score (precision/recall of boundary pixels).
    /// </summary>
    public double BoundaryF1 { get; set; }

    /// <summary>
    /// Trimap accuracy (accuracy in the uncertain boundary region).
    /// </summary>
    public double TrimapAccuracy { get; set; }

    #endregion

    #region Computational Metrics

    /// <summary>
    /// Inference time per image/frame.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Frames per second.
    /// </summary>
    public double FPS { get; set; }

    /// <summary>
    /// Peak GPU memory usage in MB.
    /// </summary>
    public double PeakGpuMemoryMB { get; set; }

    /// <summary>
    /// Number of model parameters in millions.
    /// </summary>
    public double ParameterCountM { get; set; }

    /// <summary>
    /// Floating point operations in GFLOPs.
    /// </summary>
    public double GFLOPs { get; set; }

    #endregion

    #region Dataset Info

    /// <summary>
    /// Name of the evaluation dataset (e.g., "COCO", "ADE20K", "Cityscapes", "DAVIS").
    /// </summary>
    public string? DatasetName { get; set; }

    /// <summary>
    /// Number of images/frames evaluated.
    /// </summary>
    public int NumSamples { get; set; }

    /// <summary>
    /// Number of classes in the evaluation.
    /// </summary>
    public int NumClasses { get; set; }

    #endregion
}
