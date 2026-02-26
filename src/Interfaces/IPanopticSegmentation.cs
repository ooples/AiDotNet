namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for panoptic segmentation models that unify semantic and instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Panoptic segmentation provides a complete scene understanding by combining:
/// - Semantic segmentation for "stuff" classes (sky, road, grass — amorphous regions)
/// - Instance segmentation for "things" classes (car, person, dog — countable objects)
/// Every pixel receives both a class label and an instance ID (for thing classes).
/// </para>
/// <para>
/// <b>For Beginners:</b> Panoptic segmentation gives you the most complete picture.
///
/// For a street scene, you get:
/// - "Stuff" regions: road (no instance), sky (no instance), building (no instance)
/// - "Things" instances: person #1, person #2, car #1, car #2, bicycle #1
///
/// This is useful when you need to know both "what is everywhere" AND "how many of each thing".
///
/// Models implementing this interface:
/// - Mask2Former (CVPR 2022, 57.8 PQ on COCO)
/// - kMaX-DeepLab (CVPR 2023, cross-attention as clustering)
/// - ODISE (CVPR 2023, diffusion features)
/// - OneFormer (CVPR 2023, text-conditioned)
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("PanopticSegmentation")]
public interface IPanopticSegmentation<T> : ISegmentationModel<T>
{
    /// <summary>
    /// Gets the number of "stuff" classes (amorphous regions like sky, road).
    /// </summary>
    int NumStuffClasses { get; }

    /// <summary>
    /// Gets the number of "thing" classes (countable objects like car, person).
    /// </summary>
    int NumThingClasses { get; }

    /// <summary>
    /// Performs panoptic segmentation on an image.
    /// </summary>
    /// <param name="image">Input image tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>Panoptic segmentation result with per-pixel class and instance labels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a map where each pixel has both a class label
    /// (what it is) and an instance ID (which specific object it belongs to, for countable things).
    /// Stuff classes share a single ID per class since they don't have individual instances.
    /// </para>
    /// </remarks>
    PanopticSegmentationResult<T> SegmentPanoptic(Tensor<T> image);
}

/// <summary>
/// Result of panoptic segmentation containing both semantic and instance information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PanopticSegmentationResult<T>
{
    /// <summary>
    /// Per-pixel semantic class labels [H, W].
    /// </summary>
    public Tensor<T> SemanticMap { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Per-pixel instance IDs [H, W]. Stuff classes have ID 0; thing instances have unique positive IDs.
    /// </summary>
    public Tensor<T> InstanceMap { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// Combined panoptic ID map [H, W] encoded as classId * maxInstances + instanceId.
    /// </summary>
    public Tensor<T> PanopticMap { get; set; } = Tensor<T>.Empty();

    /// <summary>
    /// List of detected thing segments with metadata.
    /// </summary>
    public List<PanopticSegment<T>> Segments { get; set; } = [];
}

/// <summary>
/// A single segment in a panoptic segmentation result.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class PanopticSegment<T>
{
    /// <summary>
    /// Unique segment ID.
    /// </summary>
    public int SegmentId { get; set; }

    /// <summary>
    /// Class ID of this segment.
    /// </summary>
    public int ClassId { get; set; }

    /// <summary>
    /// Whether this segment is a "thing" (true) or "stuff" (false).
    /// </summary>
    public bool IsThing { get; set; }

    /// <summary>
    /// Confidence score in [0, 1].
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Area of this segment in pixels.
    /// </summary>
    public int Area { get; set; }

    /// <summary>
    /// Binary mask for this segment [H, W].
    /// </summary>
    public Tensor<T> Mask { get; set; } = Tensor<T>.Empty();
}
