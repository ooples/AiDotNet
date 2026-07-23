namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Output for 3D point cloud segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Point cloud segmentation labels each 3D point (from LiDAR, depth
/// cameras, etc.) with a class. This output contains per-point labels, confidences, and
/// optional per-point features for downstream tasks like scene understanding.
/// </para>
/// </remarks>
public class PointCloudSegmentationOutput<T>
{
    /// <summary>
    /// Per-point class labels [N].
    /// </summary>
    public Tensor<T>? Labels { get; set; }

    /// <summary>
    /// Per-point class logits [N, numClasses].
    /// </summary>
    public Tensor<T>? Logits { get; set; }

    /// <summary>
    /// Per-point confidence scores [N].
    /// </summary>
    public Tensor<T>? Confidences { get; set; }

    /// <summary>
    /// Per-point learned feature vectors [N, featureDim] from the model's encoder.
    /// Useful for downstream tasks like clustering or retrieval.
    /// </summary>
    public Tensor<T>? Features { get; set; }

    /// <summary>
    /// Per-point instance IDs [N] for instance segmentation (if supported).
    /// </summary>
    public Tensor<T>? InstanceIds { get; set; }

    /// <summary>
    /// Number of points in the input cloud.
    /// </summary>
    public int NumPoints { get; set; }

    /// <summary>
    /// Number of classes.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Per-class point counts.
    /// </summary>
    public Dictionary<int, int> ClassPointCounts { get; set; } = [];

    /// <summary>
    /// Per-class names (if available).
    /// </summary>
    public Dictionary<int, string>? ClassNames { get; set; }

    /// <summary>
    /// Inference time.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Bounding boxes for each detected instance [numInstances, 6] as (x1, y1, z1, x2, y2, z2).
    /// </summary>
    public Tensor<T>? InstanceBoundingBoxes { get; set; }
}
