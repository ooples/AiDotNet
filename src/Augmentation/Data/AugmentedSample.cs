using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Data;

/// <summary>
/// Represents a sample with its data and associated spatial targets.
/// </summary>
/// <remarks>
/// <para>
/// An augmented sample bundles together the data (e.g., image) with all its
/// associated annotations that need to be transformed together when spatial
/// augmentations are applied.
/// </para>
/// <para><b>For Beginners:</b> When you rotate an image, you also need to rotate
/// any bounding boxes, keypoints, or segmentation masks associated with it.
/// This class keeps all these elements together so they transform correctly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type (e.g., ImageTensor).</typeparam>
public class AugmentedSample<T, TData>
{
    /// <summary>
    /// Gets or sets the primary data (e.g., image).
    /// </summary>
    public TData Data { get; set; }

    /// <summary>
    /// Gets or sets the label(s) for this sample.
    /// </summary>
    public Vector<T>? Labels { get; set; }

    /// <summary>
    /// Gets or sets the bounding boxes for object detection.
    /// </summary>
    public List<BoundingBox<T>>? BoundingBoxes { get; set; }

    /// <summary>
    /// Gets or sets the keypoints for pose estimation.
    /// </summary>
    public List<Keypoint<T>>? Keypoints { get; set; }

    /// <summary>
    /// Gets or sets the segmentation masks.
    /// </summary>
    public List<SegmentationMask<T>>? Masks { get; set; }

    /// <summary>
    /// Gets or sets additional metadata for this sample.
    /// </summary>
    public IDictionary<string, object> Metadata { get; set; }

    /// <summary>
    /// Creates a new augmented sample with only data.
    /// </summary>
    /// <param name="data">The primary data.</param>
    public AugmentedSample(TData data)
    {
        Data = data;
        Metadata = new Dictionary<string, object>();
    }

    /// <summary>
    /// Creates a new augmented sample with data and labels.
    /// </summary>
    /// <param name="data">The primary data.</param>
    /// <param name="labels">The labels.</param>
    public AugmentedSample(TData data, Vector<T>? labels)
        : this(data)
    {
        Labels = labels;
    }

    /// <summary>
    /// Creates a deep copy of this sample.
    /// </summary>
    /// <returns>A new sample with copied data.</returns>
    public AugmentedSample<T, TData> Clone()
    {
        var clone = new AugmentedSample<T, TData>(Data)
        {
            Labels = Labels,
            Metadata = new Dictionary<string, object>(Metadata)
        };

        if (BoundingBoxes is not null)
        {
            clone.BoundingBoxes = BoundingBoxes.Select(b => b.Clone()).ToList();
        }

        if (Keypoints is not null)
        {
            clone.Keypoints = Keypoints.Select(k => k.Clone()).ToList();
        }

        if (Masks is not null)
        {
            clone.Masks = Masks.Select(m => m.Clone()).ToList();
        }

        return clone;
    }

    /// <summary>
    /// Gets whether this sample has any bounding boxes.
    /// </summary>
    public bool HasBoundingBoxes => BoundingBoxes is not null && BoundingBoxes.Count > 0;

    /// <summary>
    /// Gets whether this sample has any keypoints.
    /// </summary>
    public bool HasKeypoints => Keypoints is not null && Keypoints.Count > 0;

    /// <summary>
    /// Gets whether this sample has any segmentation masks.
    /// </summary>
    public bool HasMasks => Masks is not null && Masks.Count > 0;

    /// <summary>
    /// Gets whether this sample has any spatial targets.
    /// </summary>
    public bool HasSpatialTargets => HasBoundingBoxes || HasKeypoints || HasMasks;
}
