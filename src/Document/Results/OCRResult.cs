namespace AiDotNet.Document;

/// <summary>
/// Represents the result of OCR (Optical Character Recognition) processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> OCR converts images of text into machine-readable text.
/// This result class contains the recognized text along with position information
/// for each word, line, and block of text.
/// </para>
/// </remarks>
public class OCRResult<T>
{
    /// <summary>
    /// Gets the full recognized text from the document.
    /// </summary>
    public string FullText { get; init; } = string.Empty;

    /// <summary>
    /// Gets the recognized words with their positions and confidence.
    /// </summary>
    public IReadOnlyList<OCRWord<T>> Words { get; init; } = [];

    /// <summary>
    /// Gets the recognized lines of text.
    /// </summary>
    public IReadOnlyList<OCRLine<T>> Lines { get; init; } = [];

    /// <summary>
    /// Gets the text blocks (paragraphs or regions).
    /// </summary>
    public IReadOnlyList<OCRBlock<T>> Blocks { get; init; } = [];

    /// <summary>
    /// Gets the average confidence across all recognized text.
    /// </summary>
    public T AverageConfidence { get; init; } = default!;

    /// <summary>
    /// Gets the detected language code (ISO 639-1, e.g., "en", "zh").
    /// </summary>
    public string? DetectedLanguage { get; init; }

    /// <summary>
    /// Gets processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; init; }

    /// <summary>
    /// Gets whether the document appears to be rotated or skewed.
    /// </summary>
    public bool RequiresDeskewing { get; init; }

    /// <summary>
    /// Gets the detected rotation angle in degrees if applicable.
    /// </summary>
    public double? RotationAngle { get; init; }
}

/// <summary>
/// Represents a single recognized word.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OCRWord<T>
{
    /// <summary>
    /// Gets the recognized text.
    /// </summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Gets the bounding box as [x1, y1, x2, y2].
    /// </summary>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets the confidence score (0-1).
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the line index this word belongs to.
    /// </summary>
    public int LineIndex { get; init; }

    /// <summary>
    /// Gets the block index this word belongs to.
    /// </summary>
    public int BlockIndex { get; init; }
}

/// <summary>
/// Represents a line of recognized text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OCRLine<T>
{
    /// <summary>
    /// Gets the full text of the line.
    /// </summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Gets the bounding box as [x1, y1, x2, y2].
    /// </summary>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets the words in this line.
    /// </summary>
    public IReadOnlyList<OCRWord<T>> Words { get; init; } = [];

    /// <summary>
    /// Gets the average confidence for words in this line.
    /// </summary>
    public T AverageConfidence { get; init; } = default!;

    /// <summary>
    /// Gets the block index this line belongs to.
    /// </summary>
    public int BlockIndex { get; init; }
}

/// <summary>
/// Represents a block (paragraph or region) of recognized text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OCRBlock<T>
{
    /// <summary>
    /// Gets the full text of the block.
    /// </summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Gets the bounding box as [x1, y1, x2, y2].
    /// </summary>
    public Vector<T> BoundingBox { get; init; } = Vector<T>.Empty();

    /// <summary>
    /// Gets the lines in this block.
    /// </summary>
    public IReadOnlyList<OCRLine<T>> Lines { get; init; } = [];

    /// <summary>
    /// Gets the average confidence for text in this block.
    /// </summary>
    public T AverageConfidence { get; init; } = default!;

    /// <summary>
    /// Gets the detected block type (text, table, figure, etc.).
    /// </summary>
    public LayoutElementType BlockType { get; init; } = LayoutElementType.Text;
}
