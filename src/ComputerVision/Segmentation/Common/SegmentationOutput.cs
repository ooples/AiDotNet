namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Unified output type for all segmentation tasks, combining semantic, instance, and panoptic results.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This is the universal result type returned by segmentation models.
/// Depending on the model, different fields will be populated:
/// - Semantic models fill <see cref="ClassMap"/> and <see cref="ClassProbabilities"/>
/// - Instance models fill <see cref="InstanceMasks"/> and <see cref="InstanceClasses"/>
/// - Panoptic models fill both semantic and instance fields plus <see cref="PanopticMap"/>
///
/// Use <see cref="TaskType"/> to determine which fields are available.
/// </para>
/// </remarks>
public class SegmentationOutput<T>
{
    #region Core Output

    /// <summary>
    /// The segmentation task that produced this output.
    /// </summary>
    public SegmentationTaskType TaskType { get; set; }

    /// <summary>
    /// Per-pixel class label map [H, W]. Each value is a class index.
    /// Populated for semantic and panoptic segmentation.
    /// </summary>
    public Tensor<T>? ClassMap { get; set; }

    /// <summary>
    /// Per-pixel class probability map [numClasses, H, W]. Values in [0, 1].
    /// Populated for semantic segmentation models that output probabilities.
    /// </summary>
    public Tensor<T>? ClassProbabilities { get; set; }

    /// <summary>
    /// Raw logits output [numClasses, H, W] before softmax/argmax.
    /// Useful for custom post-processing or ensembling.
    /// </summary>
    public Tensor<T>? Logits { get; set; }

    #endregion

    #region Instance Output

    /// <summary>
    /// Per-instance binary masks [numInstances, H, W].
    /// Populated for instance and panoptic segmentation.
    /// </summary>
    public Tensor<T>? InstanceMasks { get; set; }

    /// <summary>
    /// Class ID for each detected instance [numInstances].
    /// </summary>
    public int[]? InstanceClasses { get; set; }

    /// <summary>
    /// Confidence score for each detected instance [numInstances].
    /// </summary>
    public T[]? InstanceScores { get; set; }

    /// <summary>
    /// Bounding boxes for each instance [numInstances, 4] as (x1, y1, x2, y2).
    /// </summary>
    public Tensor<T>? InstanceBoxes { get; set; }

    #endregion

    #region Panoptic Output

    /// <summary>
    /// Combined panoptic ID map [H, W] encoded as classId * 1000 + instanceId.
    /// Populated for panoptic segmentation.
    /// </summary>
    public Tensor<T>? PanopticMap { get; set; }

    /// <summary>
    /// Per-pixel instance ID map [H, W]. Stuff classes have ID 0, things have unique positive IDs.
    /// </summary>
    public Tensor<T>? InstanceIdMap { get; set; }

    /// <summary>
    /// Panoptic segment metadata.
    /// </summary>
    public List<SegmentInfo<T>>? Segments { get; set; }

    #endregion

    #region Metadata

    /// <summary>
    /// Number of classes in the output.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Number of detected instances, derived from InstanceMasks (preferred) or InstanceClasses.
    /// </summary>
    public int NumInstances => InstanceMasks?.Shape[0] ?? InstanceClasses?.Length ?? 0;

    /// <summary>
    /// Input image height.
    /// </summary>
    public int ImageHeight { get; set; }

    /// <summary>
    /// Input image width.
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Time taken for inference.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Class name labels (if available).
    /// </summary>
    public string[]? ClassNames { get; set; }

    #endregion
}

/// <summary>
/// Type of segmentation task.
/// </summary>
public enum SegmentationTaskType
{
    /// <summary>Per-pixel class labels without instance distinction.</summary>
    Semantic,
    /// <summary>Individual object detection with pixel masks.</summary>
    Instance,
    /// <summary>Combined semantic + instance segmentation.</summary>
    Panoptic,
    /// <summary>Interactive promptable segmentation.</summary>
    Promptable,
    /// <summary>Video object segmentation with tracking.</summary>
    Video,
    /// <summary>Medical image/volume segmentation.</summary>
    Medical,
    /// <summary>3D point cloud segmentation.</summary>
    PointCloud,
    /// <summary>Open-vocabulary text-guided segmentation.</summary>
    OpenVocabulary,
    /// <summary>Natural language referring segmentation.</summary>
    Referring
}

/// <summary>
/// Metadata for a single segment in panoptic/instance output.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
public class SegmentInfo<T>
{
    /// <summary>Unique segment ID.</summary>
    public int Id { get; set; }

    /// <summary>Class ID of this segment.</summary>
    public int ClassId { get; set; }

    /// <summary>Class name (if available).</summary>
    public string? ClassName { get; set; }

    /// <summary>Whether this is a "thing" (true) or "stuff" (false) segment.</summary>
    public bool IsThing { get; set; }

    /// <summary>Confidence score in [0, 1].</summary>
    public T Score { get; set; } = default!;

    /// <summary>Area of this segment in pixels.</summary>
    public int Area { get; set; }

    /// <summary>Bounding box [x1, y1, x2, y2] in pixel coordinates (for thing segments).</summary>
    public T[]? BoundingBox { get; set; }

    /// <summary>Centroid (x, y) of the segment.</summary>
    public (T X, T Y)? Centroid { get; set; }
}
