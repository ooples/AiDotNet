namespace AiDotNet.Document;

/// <summary>
/// Represents the result of page segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Page segmentation divides a document page into different
/// types of regions (paragraphs, figures, tables, etc.). This result contains
/// all detected regions with their types and locations.
/// </para>
/// </remarks>
public class PageSegmentationResult<T>
{
    /// <summary>
    /// Gets the detected document regions.
    /// </summary>
    public IReadOnlyList<DocumentRegion<T>> Regions { get; init; } = [];

    /// <summary>
    /// Gets the pixel-level segmentation mask (class index per pixel).
    /// </summary>
    public Tensor<T>? SegmentationMask { get; init; }

    /// <summary>
    /// Gets the class probabilities for each pixel (shape: [height, width, num_classes]).
    /// </summary>
    public Tensor<T>? ClassProbabilities { get; init; }

    /// <summary>
    /// Gets the processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; init; }

    /// <summary>
    /// Gets the reading order of text regions (indices into Regions list).
    /// </summary>
    public IReadOnlyList<int> ReadingOrder { get; init; } = [];

    /// <summary>
    /// Gets the total number of detected regions.
    /// </summary>
    public int RegionCount => Regions.Count;
}

/// <summary>
/// Represents a detected document region.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class DocumentRegion<T>
{
    /// <summary>
    /// Gets the type of document region.
    /// </summary>
    public DocumentRegionType RegionType { get; init; }

    /// <summary>
    /// Gets the bounding box as [x1, y1, x2, y2].
    /// </summary>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets the polygon points for non-rectangular regions.
    /// </summary>
    public IReadOnlyList<Vector<T>>? PolygonPoints { get; init; }

    /// <summary>
    /// Gets the instance segmentation mask for this region (if available).
    /// </summary>
    public Tensor<T>? InstanceMask { get; init; }

    /// <summary>
    /// Gets the detection confidence score (0-1).
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the confidence as a double value.
    /// </summary>
    public double ConfidenceValue { get; init; }

    /// <summary>
    /// Gets the region index.
    /// </summary>
    public int Index { get; init; }

    /// <summary>
    /// Gets the reading order position (lower = earlier in reading order).
    /// </summary>
    public int ReadingOrderPosition { get; init; }
}

/// <summary>
/// Types of document regions that can be detected by page segmentation.
/// </summary>
public enum DocumentRegionType
{
    /// <summary>
    /// Regular text paragraph.
    /// </summary>
    Paragraph = 0,

    /// <summary>
    /// Document title or heading.
    /// </summary>
    Title = 1,

    /// <summary>
    /// Bulleted or numbered list.
    /// </summary>
    List = 2,

    /// <summary>
    /// Table content.
    /// </summary>
    Table = 3,

    /// <summary>
    /// Figure or image.
    /// </summary>
    Figure = 4,

    /// <summary>
    /// Figure or table caption.
    /// </summary>
    Caption = 5,

    /// <summary>
    /// Page header.
    /// </summary>
    Header = 6,

    /// <summary>
    /// Page footer.
    /// </summary>
    Footer = 7,

    /// <summary>
    /// Mathematical equation or formula.
    /// </summary>
    Equation = 8,

    /// <summary>
    /// Footnote or endnote.
    /// </summary>
    Footnote = 9,

    /// <summary>
    /// Abstract section.
    /// </summary>
    Abstract = 10,

    /// <summary>
    /// Reference or bibliography entry.
    /// </summary>
    Reference = 11,

    /// <summary>
    /// Author information.
    /// </summary>
    Author = 12,

    /// <summary>
    /// Section heading.
    /// </summary>
    Section = 13,

    /// <summary>
    /// Unknown or other region type.
    /// </summary>
    Other = 99
}
