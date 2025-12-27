namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for object detection models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Object detection finds and locates objects in images.
/// This class configures how the detection model works, including:
/// - Which architecture to use (YOLO, DETR, Faster R-CNN, etc.)
/// - Model size (smaller = faster, larger = more accurate)
/// - Detection thresholds (how confident the model must be)
/// </para>
/// </remarks>
public class ObjectDetectionOptions<T>
{
    /// <summary>
    /// The detection architecture to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Different architectures have different trade-offs:
    /// - YOLO: Very fast, good accuracy, great for real-time applications
    /// - DETR: Transformer-based, no anchors, cleaner but slower
    /// - Faster R-CNN: Two-stage, highest accuracy but slower
    /// </para>
    /// </remarks>
    public DetectionArchitecture Architecture { get; set; } = DetectionArchitecture.YOLOv8;

    /// <summary>
    /// The model size variant.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Larger models are more accurate but slower:
    /// - Nano: Fastest, lowest accuracy, good for edge devices
    /// - Small: Fast with reasonable accuracy
    /// - Medium: Balanced speed and accuracy (recommended)
    /// - Large: High accuracy, slower inference
    /// - XLarge: Highest accuracy, slowest
    /// </para>
    /// </remarks>
    public ModelSize Size { get; set; } = ModelSize.Medium;

    /// <summary>
    /// The backbone network type for feature extraction.
    /// </summary>
    public BackboneType Backbone { get; set; } = BackboneType.CSPDarknet;

    /// <summary>
    /// The neck architecture for multi-scale feature fusion.
    /// </summary>
    public NeckType Neck { get; set; } = NeckType.PANet;

    /// <summary>
    /// Number of object classes to detect.
    /// </summary>
    /// <remarks>
    /// <para>Default is 80 (COCO dataset classes).</para>
    /// </remarks>
    public int NumClasses { get; set; } = 80;

    /// <summary>
    /// Minimum confidence score for a detection to be kept.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher values mean fewer but more confident detections.
    /// Typical range: 0.25 to 0.5. Start with 0.25 and increase if you get too many false positives.</para>
    /// </remarks>
    public double ConfidenceThreshold { get; set; } = 0.25;

    /// <summary>
    /// IoU threshold for Non-Maximum Suppression (NMS).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> NMS removes duplicate detections of the same object.
    /// Lower values are more aggressive at removing overlapping boxes.
    /// Typical range: 0.4 to 0.6.</para>
    /// </remarks>
    public double NmsThreshold { get; set; } = 0.45;

    /// <summary>
    /// Maximum number of detections to return per image.
    /// </summary>
    public int MaxDetections { get; set; } = 300;

    /// <summary>
    /// Input image size [height, width] the model expects.
    /// </summary>
    /// <remarks>
    /// <para>Images will be resized to this size before detection.
    /// Common sizes: 320 (fast), 640 (balanced), 1280 (high accuracy).</para>
    /// </remarks>
    public int[] InputSize { get; set; } = new[] { 640, 640 };

    /// <summary>
    /// Whether to use multi-scale inference for better accuracy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Multi-scale runs detection at multiple resolutions
    /// and combines results. More accurate but 2-3x slower.</para>
    /// </remarks>
    public bool UseMultiScale { get; set; } = false;

    /// <summary>
    /// Whether to use pre-trained weights (COCO dataset).
    /// </summary>
    public bool UsePretrained { get; set; } = true;

    /// <summary>
    /// Custom URL to download weights from.
    /// </summary>
    /// <remarks>
    /// <para>If null, uses the default weights URL for the selected architecture and size.</para>
    /// </remarks>
    public string? WeightsUrl { get; set; }

    /// <summary>
    /// Whether to freeze the backbone during fine-tuning.
    /// </summary>
    public bool FreezeBackbone { get; set; } = false;

    /// <summary>
    /// Class names for the detection classes.
    /// </summary>
    /// <remarks>
    /// <para>If null, uses COCO class names by default.</para>
    /// </remarks>
    public string[]? ClassNames { get; set; }

    /// <summary>
    /// Random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; } = 42;
}

/// <summary>
/// Detection model architectures.
/// </summary>
public enum DetectionArchitecture
{
    /// <summary>YOLOv8 - Fast and accurate single-stage detector.</summary>
    YOLOv8,
    /// <summary>YOLOv9 - Improved YOLOv8 with programmable gradient information.</summary>
    YOLOv9,
    /// <summary>YOLOv10 - End-to-end real-time detection with NMS-free training.</summary>
    YOLOv10,
    /// <summary>YOLOv11 - Latest YOLO with enhanced feature extraction.</summary>
    YOLOv11,
    /// <summary>DETR - DEtection TRansformer, end-to-end object detection.</summary>
    DETR,
    /// <summary>DINO - DETR with Improved deNoising anchor boxes.</summary>
    DINO,
    /// <summary>RT-DETR - Real-Time DETR for efficient transformer detection.</summary>
    RTDETR,
    /// <summary>Faster R-CNN - Two-stage detector with region proposals.</summary>
    FasterRCNN,
    /// <summary>Cascade R-CNN - Multi-stage R-CNN with progressive refinement.</summary>
    CascadeRCNN
}

/// <summary>
/// Model size variants.
/// </summary>
public enum ModelSize
{
    /// <summary>Nano - Smallest and fastest model.</summary>
    Nano,
    /// <summary>Small - Compact model with good speed.</summary>
    Small,
    /// <summary>Medium - Balanced model (recommended).</summary>
    Medium,
    /// <summary>Large - High accuracy model.</summary>
    Large,
    /// <summary>XLarge - Highest accuracy, largest model.</summary>
    XLarge
}

/// <summary>
/// Backbone network types for feature extraction.
/// </summary>
public enum BackboneType
{
    /// <summary>CSPDarknet - Cross Stage Partial Darknet, used in YOLO.</summary>
    CSPDarknet,
    /// <summary>ResNet50 - 50-layer residual network.</summary>
    ResNet50,
    /// <summary>ResNet101 - 101-layer residual network.</summary>
    ResNet101,
    /// <summary>SwinT - Swin Transformer Tiny.</summary>
    SwinT,
    /// <summary>SwinS - Swin Transformer Small.</summary>
    SwinS,
    /// <summary>SwinB - Swin Transformer Base.</summary>
    SwinB,
    /// <summary>EfficientNetB0 - Efficient neural network baseline.</summary>
    EfficientNetB0,
    /// <summary>EfficientNetB4 - Larger EfficientNet variant.</summary>
    EfficientNetB4
}

/// <summary>
/// Neck architecture types for multi-scale feature fusion.
/// </summary>
public enum NeckType
{
    /// <summary>FPN - Feature Pyramid Network.</summary>
    FPN,
    /// <summary>PANet - Path Aggregation Network.</summary>
    PANet,
    /// <summary>BiFPN - Bidirectional Feature Pyramid Network.</summary>
    BiFPN
}

/// <summary>
/// NMS (Non-Maximum Suppression) algorithm variants.
/// </summary>
public enum NmsType
{
    /// <summary>Standard hard NMS - removes overlapping boxes.</summary>
    Standard,
    /// <summary>Soft-NMS - reduces confidence of overlapping boxes instead of removing.</summary>
    Soft,
    /// <summary>DIoU-NMS - uses Distance-IoU for better localization.</summary>
    DIoU
}
