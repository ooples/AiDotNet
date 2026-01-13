namespace AiDotNet.Document;

/// <summary>
/// Result of reading order detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ReadingOrderResult<T>
{
    /// <summary>
    /// Gets or sets the ordered elements.
    /// </summary>
    public IList<OrderedElement<T>> OrderedElements { get; init; } = [];

    /// <summary>
    /// Gets or sets the confidence score as generic type.
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets or sets the confidence score as double.
    /// </summary>
    public double ConfidenceValue { get; init; }

    /// <summary>
    /// Gets or sets the processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; init; }
}

/// <summary>
/// Represents an element with reading order information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class OrderedElement<T>
{
    /// <summary>
    /// Gets or sets the element index.
    /// </summary>
    public int ElementIndex { get; init; }

    /// <summary>
    /// Gets or sets the reading order position.
    /// </summary>
    public int ReadingOrderPosition { get; init; }

    /// <summary>
    /// Gets or sets the confidence score as generic type.
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets or sets the confidence score as double.
    /// </summary>
    public double ConfidenceValue { get; init; }
}
