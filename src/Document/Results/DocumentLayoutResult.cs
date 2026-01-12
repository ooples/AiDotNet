namespace AiDotNet.Document;

/// <summary>
/// Represents the result of document layout detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Layout detection identifies different parts of a document
/// (text blocks, tables, figures, etc.) and their locations. This result class
/// contains all the detected regions with their bounding boxes and types.
/// </para>
/// </remarks>
public class DocumentLayoutResult<T>
{
    /// <summary>
    /// Gets the detected layout regions.
    /// </summary>
    public IReadOnlyList<LayoutRegion<T>> Regions { get; init; } = [];

    /// <summary>
    /// Gets the total number of detected regions.
    /// </summary>
    public int TotalRegions => Regions.Count;

    /// <summary>
    /// Gets regions filtered by element type.
    /// </summary>
    /// <param name="elementType">The type of elements to retrieve.</param>
    /// <returns>Regions matching the specified type.</returns>
    public IEnumerable<LayoutRegion<T>> GetRegionsByType(LayoutElementType elementType)
    {
        return Regions.Where(r => r.ElementType == elementType);
    }

    /// <summary>
    /// Gets regions with confidence above a threshold.
    /// </summary>
    /// <param name="threshold">Minimum confidence (0-1).</param>
    /// <returns>Regions with confidence above the threshold.</returns>
    public IEnumerable<LayoutRegion<T>> GetHighConfidenceRegions(double threshold)
    {
        return Regions.Where(r => r.ConfidenceValue >= threshold);
    }

    /// <summary>
    /// Gets the reading order of text regions (if detected).
    /// </summary>
    public IReadOnlyList<int>? ReadingOrder { get; init; }

    /// <summary>
    /// Gets processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; init; }
}

/// <summary>
/// Represents a single detected layout region in a document.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class LayoutRegion<T>
{
    /// <summary>
    /// Gets the type of layout element.
    /// </summary>
    public LayoutElementType ElementType { get; init; }

    /// <summary>
    /// Gets the bounding box as [x1, y1, x2, y2] in pixels or normalized coordinates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// x1, y1: Top-left corner
    /// x2, y2: Bottom-right corner
    /// </para>
    /// </remarks>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets the confidence score for this detection (0-1).
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the confidence as a double value for comparison operations.        
    /// </summary>
    public double ConfidenceValue { get; init; }

    /// <summary>
    /// Gets the region index in the original detection order.
    /// </summary>
    public int Index { get; init; }

    /// <summary>
    /// Gets the text content if available (requires OCR).
    /// </summary>
    public string? TextContent { get; init; }

    /// <summary>
    /// Gets the parent region index if this is a nested element.
    /// </summary>
    public int? ParentIndex { get; init; }

    /// <summary>
    /// Gets the classification of confidence level.
    /// </summary>
    public ConfidenceLevel ConfidenceLevel
    {
        get
        {
            return ConfidenceValue switch
            {
                < 0.25 => ConfidenceLevel.VeryLow,
                < 0.50 => ConfidenceLevel.Low,
                < 0.75 => ConfidenceLevel.Medium,
                < 0.90 => ConfidenceLevel.High,
                _ => ConfidenceLevel.VeryHigh
            };
        }
    }
}
