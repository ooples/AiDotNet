namespace AiDotNet.Document;

/// <summary>
/// Represents the result of text detection in an image.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Text detection finds where text is located in an image.
/// This result contains the bounding boxes (locations) of all detected text regions,
/// but not the actual text content - that requires a text recognizer.
/// </para>
/// </remarks>
public class TextDetectionResult<T>
{
    /// <summary>
    /// Gets the detected text regions.
    /// </summary>
    public IReadOnlyList<TextRegion<T>> TextRegions { get; init; } = [];

    /// <summary>
    /// Gets the probability map showing text likelihood at each pixel.
    /// </summary>
    public Tensor<T>? ProbabilityMap { get; init; }

    /// <summary>
    /// Gets the threshold map (for models like DBNet that use adaptive thresholding).
    /// </summary>
    public Tensor<T>? ThresholdMap { get; init; }

    /// <summary>
    /// Gets the binary map (final segmentation result).
    /// </summary>
    public Tensor<T>? BinaryMap { get; init; }

    /// <summary>
    /// Gets the processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; init; }

    /// <summary>
    /// Gets the number of detected text regions.
    /// </summary>
    public int RegionCount => TextRegions.Count;
}

/// <summary>
/// Represents a detected text region in an image.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TextRegion<T>
{
    /// <summary>
    /// Gets the bounding box as [x1, y1, x2, y2] for axis-aligned rectangles.
    /// </summary>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets the polygon points for rotated/curved text (list of [x, y] coordinates).
    /// </summary>
    public IReadOnlyList<Vector<T>>? PolygonPoints { get; init; }

    /// <summary>
    /// Gets the detection confidence score (0-1).
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the confidence as a double value.
    /// </summary>
    public double ConfidenceValue { get; init; }

    /// <summary>
    /// Gets the rotation angle in degrees (if detected).
    /// </summary>
    public double? RotationAngle { get; init; }

    /// <summary>
    /// Gets the region index.
    /// </summary>
    public int Index { get; init; }

    /// <summary>
    /// Gets the cropped image of this text region (if available).
    /// </summary>
    public Tensor<T>? CroppedImage { get; init; }
}
